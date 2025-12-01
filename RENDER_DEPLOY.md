# 🚀 Deploying to Render

Follow these exact steps to deploy your Brain Tumor Detection app on Render.

## Prerequisites
*   You have a GitHub account.
*   Your code is pushed to GitHub (I have already done this for you).

## Step-by-Step Guide

1.  **Sign Up / Log In**
    *   Go to [dashboard.render.com](https://dashboard.render.com/).
    *   Sign in with your **GitHub** account.

2.  **Create New Service**
    *   Click the **"New +"** button in the top right corner.
    *   Select **"Web Service"**.

3.  **Connect Repository**
    *   You should see a list of your GitHub repositories.
    *   Find **`Brats`** in the list.
    *   Click **"Connect"**.

4.  **Configure Service**
    *   **Name**: Enter a unique name (e.g., `brain-tumor-app`).
    *   **Region**: Select the one closest to you (e.g., Singapore, Frankfurt, Oregon).
    *   **Branch**: Ensure `main` is selected.
    *   **Runtime**: Select **Docker** (This is very important!).

5.  **Instance Type**
    *   Scroll down to "Instance Type".
    *   Select **"Free"** (0.5 CPU, 512MB RAM).

6.  **Environment Variables (Optional)**
    *   You don't typically need any for this app unless you want to customize it.

7.  **Deploy**
    *   Click the **"Create Web Service"** button at the bottom.

## What Happens Next?
*   Render will start building your Docker image. You will see logs scrolling in the "Events" or "Logs" tab.
*   **Be Patient**: The first build might take 5-10 minutes because it has to install TensorFlow and other heavy libraries.
*   Once finished, you will see a green **"Live"** badge.
*   Your URL will be at the top left (e.g., `https://brain-tumor-app.onrender.com`).

## Troubleshooting
*   **Port Issue**: If the deploy succeeds but the page doesn't load, go to **Settings** -> **Container Port** and ensure it is set to **7860**. (Render usually detects this from the Dockerfile, but verify if needed).
*   **Memory**: If the build fails with "Out of Memory", you might need to try deploying again or upgrade to a paid plan, but the Free tier usually handles basic TensorFlow apps fine.
