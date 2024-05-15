# PoseDetection Demo Upload Video

This repository contains a demo application for pose detection using TensorFlow.js and MediaPipe Pose API. It demonstrates how to set up a basic web application to upload videos and process them for pose detection in real-time.

## Version

0.0.1

## Description

The demo provides an interface to upload video files and utilize a pose detection API to analyze and display the detected poses. The front end is built with simple HTML/CSS and JavaScript, leveraging various TensorFlow.js models and the MediaPipe Pose library.

## Prerequisites

This project requires Node.js version 8.9.0 or higher.

## Installation

Clone the repository and install dependencies:

```bash
git clone <repository-url>
cd posedetection_demo_upload_video
npm install
```

## Available Scripts

In the project directory, you can run:

```bash
npm start
```

Runs the app in the development mode.
Open http://localhost:3000 to view it in the browser. The page will reload if you make edits.

```bash
npm run build
```

Builds the app for production to the dist folder.
It correctly bundles in production mode and optimizes the build for the best performance.

```bash
npm run lint
```

Lints and checks the code for any issues as per configured ESLint rules.

## Development Utilities

Build Dependencies: Compiles the necessary dependencies.
Link Core: Symlinks the local TensorFlow Core package for development.
Link WebGL: Symlinks the local TensorFlow WebGL backend package for development.

## Dependencies

This project uses the following main dependencies:

- `@mediapipe/pose`
- `@tensorflow-models/pose-detection`
- TensorFlow.js modules
  - `fs-extra` for enhanced file system operations
  - `scatter-gl` for graphical display of pose data

## Dev Dependencies

Includes necessary Babel and Parcel configurations for building and running the application:

- `@babel/core`
- `@babel/preset-env`
- `parcel-bundler`
- `eslint` with Google's ESLint configuration for code linting
  Other utilities like yalc for managing local package dependencies.

## Browser Compatibility

This demo does not support the Crypto API in the browser environment.
