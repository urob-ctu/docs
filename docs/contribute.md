1. **Images:** Use the following snippet for images# How to Contribute

- [Introduction](#introduction)
- [Local Development Setup](#local-development-setup)
  - [Prerequisites](#prerequisites)
  - [Fork and Clone the Repository](#fork-and-clone-the-repository)
  - [Running the Development Server](#running-the-development-server)
- [Making and Submitting Changes](#making-and-submitting-changes)
- [Content & Style Guidelines](#content--style-guidelines)

1. **Images:** Use the following snippet for images

- [Introduction](#introduction)
- [Local Development Setup](#local-development-setup)
  - [Prerequisites](#prerequisites)
  - [Fork and Clone the Repository](#fork-and-clone-the-repository)
  - [Running the Development Server](#running-the-development-server)
- [Making and Submitting Changes](#making-and-submitting-changes)
- [Content & Style Guidelines](#content--style-guidelines)

## Introduction

The materials you see are primarily written by the teachers of the UROB course, but community contributions are highly encouraged! If you notice an error, a typo, or think something could be explained better, we invite you to help us improve the course.

Why should you contribute? First, it helps you gain a deeper understanding of the subject. Second, you are helping future students learn more effectively. Most importantly, you can **earn points for your contribution**.

For simple fixes like typos, feel free to [open an issue](https://github.com/urob-ctu/docs/issues) on GitHub. For more complex suggestions or new content, please follow the development and contribution guide below.

## Local Development Setup

To make complex changes and preview them, you must run the website on your local machine. We use Docker to provide a simple, one-command setup that works consistently for everyone.

### Prerequisites

You must have **Docker and Docker Compose** installed. This is available as a single download for all major operating systems.

*   **[Get Docker (includes Docker Compose)](https://docs.docker.com/get-docker/)**

> **Note:** These instructions use the modern `docker compose` command (with a space). If you have an older version, you may need to use `docker-compose` (with a hyphen), but we strongly recommend using an up-to-date version of Docker.

### Fork and Clone the Repository

1.  **Fork** the repository to your own GitHub account using the button at the top of the project page.

    <div align="center">
        <img src="./assets/images/fork-button.webp" width="800">
    </div>

2.  **Clone** your forked repository to your local machine and navigate into the project directory:
    ```bash
    git clone git@github.com:YOUR-USERNAME/docs.git
    cd docs
    ```

### Running the Development Server

1.  **Start the server** with a single command:
    ```bash
    docker compose up
    ```
    The first time you run this, Docker will build the necessary image, which may take a few minutes. Subsequent startups will be much faster.

2.  **View the website.** Once the server is running, you will see output like this in your terminal:
    > Server address: http://0.0.0.0:4000
    > Server running... press ctrl-c to stop.

    You can now view the live site by navigating your browser to **[http://localhost:4000](http://localhost:4000)**.

3.  **Make your changes.** The server is configured for **LiveReload**. When you save changes to any file, the site will automatically rebuild, and your browser will refresh to show your edits.

4.  **Stop the server** when you are finished by pressing **`Ctrl+C`** in the terminal.

To perform a full cleanup and remove the container, run `docker compose down`.

## Making and Submitting Changes

1.  Create a new branch for your feature or bugfix.
2.  Make your changes, previewing them with the local server as needed.
3.  Commit your changes following the [commit message guidelines](#commit-message-guidelines).
4.  Push the branch to your fork on GitHub.
5.  Open a Pull Request from your branch to the `main` branch of the original `urob-ctu/docs` repository.

## Content & Style Guidelines

To keep the materials consistent and easy to read, please follow these guidelines:

1.  **Language:** Write only in English.
2.  **Table of Contents:** Include a table of contents at the top of each page using this snippet:

    ```markdown
    # Header
    {: .no_toc }

    <details open markdown="block">
    <summary>
        Table of contents
    </summary>
    {: .text-delta }
    1. TOC
    {:toc}
    </details>
    ```

3.  **Introduction:** Include an introduction at the beginning of each page that describes its purpose.
{% raw %}
4.  **Images:** Use the following snippet for images. Always add images to the `assets/images` folder.

   ```markdown
   <div align="center">
     <img src="{{ site.baseurl }}/assets/images/IMAGE_NAME.png" width="800">
   </div>
  ```

2. **Videos:** Use the following snippet for videos:

   ```markdown
   <div align="center">
     <video src="{{ site.baseurl }}/assets/videos/spirals_relu.mp4" width="640" autoplay loop controls muted></video>
   </div>
   ```

6. **Callouts:** You can use the following callouts:
   - `{: .definition }` for definitions
   - `{: .slogan }` for important information
   - `{: .warning }` for warnings
   - `{: .important }` for important information
   - `{: .note }` for notes
  
   More about callouts [here](https://just-the-docs.com/docs/ui-components/callouts/).

7. **Commit message guidelines:** Follow these [commit message guidelines](https://gist.github.com/robertpainsi/b632364184e70900af4ab688decf6f53) when making changes.

{% endraw %}

A good rule of thumb is to look at existing pages and follow the same structure. If you can't find something, refer to the documentation of the [Just the Docs](https://just-the-docs.com/) template we use.
