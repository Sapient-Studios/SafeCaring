/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import "@tensorflow/tfjs-backend-webgl";
import "@tensorflow/tfjs-backend-webgpu";
import * as mpPose from "@mediapipe/pose";

import * as tfjsWasm from "@tensorflow/tfjs-backend-wasm";

tfjsWasm.setWasmPaths(`https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${tfjsWasm.version_wasm}/dist/`);

import * as posedetection from "@tensorflow-models/pose-detection";
import * as tf from "@tensorflow/tfjs-core";

import { setupStats } from "./stats_panel";
import { Context } from "./camera";
import { setupDatGui } from "./option_panel";
import { STATE } from "./params";
import { setBackendAndEnvFlags } from "./util";
import { RendererWebGPU } from "./renderer_webgpu";
import { RendererCanvas2d } from "./renderer_canvas2d";

let detector, camera, stats;
let startInferenceTime,
	numInferences = 0;
let inferenceTimeSum = 0,
	lastPanelUpdate = 0;
let rafId;
let renderer = null;

let useGpuRenderer = false;

const statusElement = document.getElementById("status");
const resultsDiv = document.getElementById("results");

const THRESHOLD = 0.65; // Define a suitable threshold for detecting sudden movements
const UPDATE_INTERVAL = 10; // Interval in milliseconds to update the display
const ANOMALY_DISPLAY_DURATION = 500; // 0.5 seconds to prioritize anomaly display
const lastStandardDeviationMap = new Map();

let lastUpdateTime = 0;
let lastAnomalyTime = 0;

function calculateStandardDeviation(values) {
	const mean = values.reduce((a, b) => a + b, 0) / values.length;
	const squareDiffs = values.map((value) => {
		const diff = value - mean;
		return diff * diff;
	});
	const avgSquareDiff = squareDiffs.reduce((a, b) => a + b, 0) / squareDiffs.length;
	return Math.sqrt(avgSquareDiff);
}

function calculateMean(values) {
	return values.reduce((a, b) => a + b, 0) / values.length;
}

// Main function to process new pose data
function processNewPoseData(poses) {
	const currentTime = Date.now();

	poses.forEach((pose) => {
		pose.keypoints.forEach((kp) => {
			if (["left_shoulder", "right_shoulder", "left_hip", "right_hip", "nose"].includes(kp.name) && kp.score > 0.7) {
				const vectors = [kp.x, kp.y, kp.z];
				const norms = vectors.map((vec) => Math.sqrt(vec ** 2));
				const stdDev = calculateStandardDeviation(norms);
				const mean = calculateMean(norms);

				let lastStdDev = lastStandardDeviationMap.get(kp.name) || 0;

				if (Math.abs(stdDev - lastStdDev) >= THRESHOLD) {
					console.log("Anomaly detected");
					lastStandardDeviationMap.set(kp.name, stdDev);
					lastAnomalyTime = currentTime;
					updateDisplay(stdDev, mean, "Anomaly detected");
				} else {
					lastStandardDeviationMap.set(kp.name, stdDev);
					if (currentTime - lastAnomalyTime > ANOMALY_DISPLAY_DURATION) {
						if (currentTime - lastUpdateTime > UPDATE_INTERVAL) {
							updateDisplay(stdDev, mean, "No significant movement detected");
							lastUpdateTime = currentTime;
						}
					}
				}
			}
		});
	});
}
function updateDisplay(stdDev, mean, state) {
	resultsDiv.innerHTML = `
		<p>The standard deviation is: ${stdDev.toFixed(2)}</p>
		<p>The mean of the vector is: ${mean.toFixed(2)}</p>
		<p>Pacient state: ${state}</p>
		<hr>`;
}

async function createDetector() {
	switch (STATE.model) {
		case posedetection.SupportedModels.PoseNet:
			return posedetection.createDetector(STATE.model, {
				quantBytes: 4,
				architecture: "MobileNetV1",
				outputStride: 16,
				inputResolution: { width: 500, height: 500 },
				multiplier: 0.75,
			});
		case posedetection.SupportedModels.BlazePose:
			const runtime = STATE.backend.split("-")[0];
			if (runtime === "mediapipe") {
				return posedetection.createDetector(STATE.model, {
					runtime,
					modelType: STATE.modelConfig.type,
					solutionPath: `https://cdn.jsdelivr.net/npm/@mediapipe/pose@${mpPose.VERSION}`,
				});
			} else if (runtime === "tfjs") {
				return posedetection.createDetector(STATE.model, { runtime, modelType: STATE.modelConfig.type });
			}
		case posedetection.SupportedModels.MoveNet:
			const modelType = STATE.modelConfig.type == "lightning" ? posedetection.movenet.modelType.SINGLEPOSE_LIGHTNING : posedetection.movenet.modelType.SINGLEPOSE_THUNDER;
			return posedetection.createDetector(STATE.model, { modelType });
	}
}

async function checkGuiUpdate() {
	if (STATE.isModelChanged || STATE.isFlagChanged || STATE.isBackendChanged) {
		STATE.isModelChanged = true;

		window.cancelAnimationFrame(rafId);

		detector.dispose();

		if (STATE.isFlagChanged || STATE.isBackendChanged) {
			await setBackendAndEnvFlags(STATE.flags, STATE.backend);
		}

		detector = await createDetector(STATE.model);
		STATE.isFlagChanged = false;
		STATE.isBackendChanged = false;
		STATE.isModelChanged = false;
	}
}

function beginEstimatePosesStats() {
	startInferenceTime = (performance || Date).now();
}

function endEstimatePosesStats() {
	const endInferenceTime = (performance || Date).now();
	inferenceTimeSum += endInferenceTime - startInferenceTime;
	++numInferences;

	const panelUpdateMilliseconds = 1000;
	if (endInferenceTime - lastPanelUpdate >= panelUpdateMilliseconds) {
		const averageInferenceTime = inferenceTimeSum / numInferences;
		inferenceTimeSum = 0;
		numInferences = 0;
		stats.customFpsPanel.update(1000.0 / averageInferenceTime, 120 /* maxValue */);
		lastPanelUpdate = endInferenceTime;
	}
}

async function renderResult() {
	// FPS only counts the time it takes to finish estimatePoses.
	beginEstimatePosesStats();

	const poses = await detector.estimatePoses(camera.video, { maxPoses: STATE.modelConfig.maxPoses, flipHorizontal: false });

	processNewPoseData(poses);

	const currenttimestamp = Date.now();

	// log now the current timestamp

	// can you write log to a json, write to ls,
	// get the log from an array

	// no need to write logs
	/* 
	const log = {
		timestamp: currenttimestamp,
		poses,
	};
	const logs = JSON.parse(localStorage.getItem("log")) || [];
	// apeend the new log to the logs array
	logs.push(log);
	// write the logs array to the local storage
	localStorage.setItem("log", JSON.stringify(logs)); */

	endEstimatePosesStats();

	camera.drawCtx();

	// The null check makes sure the UI is not in the middle of changing to a
	// different model. If during model change, the result is from an old
	// model, which shouldn't be rendered.
	if (poses.length > 0 && !STATE.isModelChanged) {
		camera.drawResults(poses);
	}

	const rendererParams = useGpuRenderer ? [camera.video, poses, canvasInfo, STATE.modelConfig.scoreThreshold] : [camera.video, poses, STATE.isModelChanged];
	renderer.draw(rendererParams);
}

async function updateVideo(event) {
	// Clear reference to any previous uploaded video.
	URL.revokeObjectURL(camera.video.currentSrc);
	const file = event.target.files[0];
	camera.source.src = URL.createObjectURL(file);

	// Wait for video to be loaded.
	camera.video.load();
	await new Promise((resolve) => {
		camera.video.onloadeddata = () => {
			resolve(video);
		};
	});

	const videoWidth = camera.video.videoWidth;
	const videoHeight = camera.video.videoHeight;
	// Must set below two lines, otherwise video element doesn't show.
	camera.video.width = videoWidth;
	camera.video.height = videoHeight;
	camera.canvas.width = videoWidth;
	camera.canvas.height = videoHeight;

	statusElement.innerHTML = "Video is loaded.";
}

async function runFrame() {
	await checkGuiUpdate();
	if (video.paused) {
		// video has finished.
		camera.mediaRecorder.stop();
		camera.clearCtx();
		camera.video.style.visibility = "visible";
		return;
	}
	await renderResult();
	rafId = requestAnimationFrame(runFrame);
}

async function run() {
	statusElement.innerHTML = "Warming up model.";

	// Warming up pipeline.
	const [runtime, $backend] = STATE.backend.split("-");

	if (runtime === "tfjs") {
		const warmUpTensor = tf.fill([camera.video.height, camera.video.width, 3], 0, "float32");
		await detector.estimatePoses(warmUpTensor, { maxPoses: STATE.modelConfig.maxPoses, flipHorizontal: false });
		warmUpTensor.dispose();
		statusElement.innerHTML = "Model is warmed up.";
	}

	camera.video.style.visibility = "hidden";
	video.pause();
	video.currentTime = 0;
	video.play();
	camera.mediaRecorder.start();

	await new Promise((resolve) => {
		camera.video.onseeked = () => {
			resolve(video);
		};
	});

	await runFrame();
}

async function app() {
	// Gui content will change depending on which model is in the query string.
	const urlParams = new URLSearchParams(window.location.search);
	if (!urlParams.has("model")) {
		urlParams.set("model", "blazepose");
	}

	await setupDatGui(urlParams);
	stats = setupStats();
	camera = new Context();

	await setBackendAndEnvFlags(STATE.flags, STATE.backend);
	await tf.ready();
	detector = await createDetector();

	const runButton = document.getElementById("submit");
	runButton.onclick = run;

	const uploadButton = document.getElementById("videofile");
	uploadButton.onchange = updateVideo;

	const canvas = document.getElementById("output");
	canvas.width = camera.video.width;
	canvas.height = camera.video.height;
	useGpuRenderer = urlParams.get("gpuRenderer") === "true" && isWebGPU;
	if (useGpuRenderer) {
		renderer = new RendererWebGPU(canvas, importVideo);
	} else {
		renderer = new RendererCanvas2d(canvas);
	}
}

app();
