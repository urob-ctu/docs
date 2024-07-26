---
title: How to Contribute
layout: default
nav_order: 10
mathjax: true
---

# How to Contribute
{: .no_toc }

<details open markdown="block">
  <summary>
    Table of Contents
  </summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

## Introduction

The materials you see are primarily written by the teachers of the UROB course, but you can also contribute! If you notice an error or a typo, please simply [open an issue](https://github.com/urob-ctu/urob-ctu.github.io/issues) on GitHub. If you have a more complex suggestion or an entirely new section, follow these contribution guidelines.

Why should you contribute? First, it helps you gain a better understanding of the subject. Second, it helps us improve the course. Most importantly, you can **earn points for your contribution**.


## Run the Website Locally

When making more complex changes to the website, you should run the website locally to see the results of your changes. To develop the website locally, follow these steps:

### Fork and Clone the Repository

First, fork the [repository](https://github.com/urob-ctu/docs) to your own account by clicking the button at the top of the page.

<div align="center">
    <img src="{{ site.baseurl }}/assets/images/fork-button.webp" width="800">
</div>

Then, clone the forked repository by running:

```bash
git clone https://github.com/YOUR-USERNAME/docs.git
```

Navigate to the cloned repository:

```bash
cd docs
```

### Building and Previewing the Website

The following dependencies are required to build the website locally:

- [Jekyll](https://jekyllrb.com)
- [Bundler](https://bundler.io)

Assuming you are in the cloned repository directory, run:

```bash
bundle install
```

Then run the run.sh script to preview the site locally:

```bash
bash run.sh
```

The website should automatically open in your browser. If not, you can find it at `localhost:4000`.


## Contribution Guidelines

To make the materials consistent and easy to read, follow these guidelines:

1. **Language:** Write only in English.
2. **Table of Contents:** Include a table of contents at the top of each page using the following snippet:

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

3. **Introduction:** Include an introduction at the beginning of each page that describes the purpose of the page.
{% raw %}
1. **Images:** Use the following snippet for images:
    
   ```markdown
   <div align="center">
     <img src="{{ site.baseurl }}/assets/images/IMAGE_NAME.png" width="800">
   </div>
   ```

    - The `<div align="center">` tag centers the image.
    - The `{{ site.baseurl }}` is the path to the website directory. Then add the relative path to the image in the `assets/images` folder. Always add images to the `assets/images` folder.
    - The `width="800"` is the width of the image. You can adjust it as needed.

2. **Videos:** Use the following snippet for videos:
   
   ```markdown
   <div align="center">
     <video src="{{ site.baseurl }}/assets/videos/spirals_relu.mp4" width="640" autoplay loop controls muted></video>
   </div>
   ```
    
    - The `width="640"` attribute sets the video width. You can adjust it as needed.
    - The `{{ site.baseurl }}` is the path to the website directory. Then add the relative path to the video in the `assets/videos` folder. Always add videos to the `assets/videos` folder.


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