# Deploying to Hugging Face Spaces

This guide explains how to deploy the Brain Tumor Detection System to Hugging Face Spaces using Docker.

## Prerequisites

1.  A [Hugging Face account](https://huggingface.co/join).
2.  Git installed on your machine.

## Steps

### 1. Create a New Space

1.  Go to [huggingface.co/new-space](https://huggingface.co/new-space).
2.  **Space Name**: Enter a name (e.g., `brain-tumor-detection`).
3.  **License**: Select `MIT` or `Apache 2.0`.
4.  **SDK**: Select **Docker**.
5.  **Visibility**: Public or Private.
6.  Click **Create Space**.

### 2. Upload Files to the Space

You can upload files directly via the web interface or use Git. Since you have the code locally, Git is recommended.

#### Option A: Using Git (Recommended)

1.  Clone your new Space repository (replace `YOUR_USERNAME` and `SPACE_NAME`):
    ```bash
    git clone https://huggingface.co/spaces/YOUR_USERNAME/SPACE_NAME
    ```

2.  Copy all files from your `F:\BRATS\CODE` folder into this new directory.
    *   **Important**: Do NOT copy the `.git` folder.
    *   Ensure `Dockerfile`, `requirements.txt`, `app.py`, `tumor.py`, `models/`, and `static/` are included.

3.  Push the files:
    ```bash
    cd SPACE_NAME
    git add .
    git commit -m "Initial deployment"
    git push
    ```

#### Option B: Drag and Drop

1.  In your Space's "Files" tab, click **Add file** > **Upload files**.
2.  Drag and drop all your project files (`app.py`, `Dockerfile`, `requirements.txt`, `models/`, `static/`, etc.).
3.  Commit the changes.

### 3. Configuration (Important)

The `Dockerfile` is configured to run on port **7860**, which is the default for Hugging Face Spaces.

*   **Models**: Ensure your `.keras` and `.h5` files in the `models/` directory are uploaded. If they are larger than 10MB, you must use Git LFS (Large File Storage).

    ```bash
    git lfs install
    git lfs track "*.keras" "*.h5"
    git add .gitattributes
    ```

### 4. Wait for Build

1.  Go to the **App** tab in your Space.
2.  You will see "Building...". This may take a few minutes as it installs dependencies and downloads Docker layers.
3.  Once finished, you will see "Running" and your app interface will appear!

## Troubleshooting

*   **"Runtime Error"**: Check the **Logs** tab.
*   **Memory Issues**: If the app crashes while loading models, you might need to upgrade the Space hardware (Settings > Hardware) or optimize the models.
*   **LFS Error**: If you see errors about file pointers, make sure you installed Git LFS before pushing large model files.
