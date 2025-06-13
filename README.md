# Rakib Al Fahad's Personal Website

This repository contains the source code for my personal website hosted at [rakibalfahad.github.io](https://rakibalfahad.github.io/). The site is built using Jekyll with the Minimal Mistakes theme.

## Site Structure

The website is organized into several sections:
- **Blog**: Personal thoughts and insights
- **Tutorials**: Step-by-step guides on various topics
- **Code Sharing**: Useful code snippets and examples
- **Projects**: Showcase of personal and academic projects
- **Machine Learning**: Posts related to machine learning
- **Resources**: Curated list of valuable resources

## How to Update the Site

### Prerequisites

1. Install Ruby and DevKit (for Windows):
   - Download from [RubyInstaller for Windows](https://rubyinstaller.org/downloads/)
   - Choose Ruby+Devkit version
   - Check "Add Ruby executables to your PATH" during installation
   - Run the ridk install step and select option 3

2. Install Jekyll and Bundler:
   ```
   gem install jekyll bundler
   ```

3. Install dependencies:
   ```
   bundle install
   ```

### Running Locally

To preview changes locally before pushing:
```
bundle exec jekyll serve
```
Then visit http://localhost:4000 in your browser.

## Adding New Content

### How to Add a New Blog Post

1. Create a new Markdown file in the `_posts` directory
2. Name it using the format: `YYYY-MM-DD-title.md`
3. Add the following front matter at the top:
   ```yaml
   ---
   title: "Your Post Title"
   date: YYYY-MM-DD
   categories:
     - category1
     - category2
   tags:
     - tag1
     - tag2
   excerpt: "A brief description of your post."
   header:
     image: "/images/your-header-image.jpg"
     teaser: "/images/your-teaser-image.jpg"
   ---
   ```
4. Write your content below the front matter using Markdown

### How to Add a New Tutorial

1. Create a new Markdown file in the `_tutorials` directory
2. Name it using a descriptive filename (e.g., `data-visualization-python.md`)
3. Add the following front matter at the top:
   ```yaml
   ---
   title: "Your Tutorial Title"
   date: YYYY-MM-DD
   categories:
     - category1
     - category2
   tags:
     - tag1
     - tag2
   header:
     image: "/images/your-header-image.jpg"
     teaser: "/images/your-teaser-image.jpg"
   excerpt: "A brief description of your tutorial."
   ---
   ```
4. Write your tutorial content using Markdown
5. For code blocks, use triple backticks with the language name:
   ```
   ```python
   # Your Python code here
   ```
   ```

### How to Add a New Code Snippet

1. Create a new Markdown file in the `_code` directory
2. Add appropriate front matter similar to tutorials
3. Organize your code snippets using Markdown headings
4. Use code blocks with syntax highlighting

### How to Add a New Project

1. Create a new Markdown file in the `_projects` directory
2. Add appropriate front matter
3. Describe your project, technologies used, and outcomes
4. Include links to GitHub repositories or live demos if available

## Adding Images and Plots

### Where to Store Images

- Place all images in the `images` directory
- For better organization, create subdirectories for different categories:
  - `images/blog/`
  - `images/tutorials/`
  - `images/projects/`
  - etc.

### How to Add Images to Posts

1. Add an image to the appropriate directory
2. Reference it in your Markdown file:
   ```markdown
   ![Image Description](/images/path-to-image.jpg)
   ```

3. For more control, use HTML:
   ```html
   <figure>
     <img src="/images/path-to-image.jpg" alt="Description">
     <figcaption>Caption for the image</figcaption>
   </figure>
   ```

### Adding Plots and Charts

1. Generate your plots using your preferred tool (matplotlib, seaborn, etc.)
2. Save the plot as an image
3. Add the image to the appropriate directory
4. Reference it in your Markdown

Or embed interactive plots:

1. Use libraries that support HTML output (Plotly, Bokeh)
2. Export the plot as HTML
3. Insert the HTML code directly in your Markdown file

## Customizing Your Site

### Changing Profile Picture

1. Replace the profile picture file at `/images/bio-pic-2.jpg` with your new image
   - Keep the same filename or update the reference in `_config.yml`
2. If you want to change the reference:
   - Open `_config.yml`
   - Find the `author` section
   - Update the `avatar` property:
     ```yaml
     author:
       avatar: "/images/your-new-profile-pic.jpg"
     ```

### Changing Banner Images

1. For the homepage banner:
   - Replace `/images/waterfront.jpg` or
   - Update the reference in `index.html`:
     ```yaml
     header:
       image: "/images/your-new-banner.jpg"
     ```

2. For page-specific banners:
   - Update the front matter in the respective page files:
     ```yaml
     header:
       image: "/images/your-new-banner.jpg"
     ```

### Changing Site Theme and Colors

1. Open `_config.yml`
2. Find the `minimal_mistakes_skin` property
3. Change it to one of the available options:
   - "default"
   - "air"
   - "aqua"
   - "contrast"
   - "dark"
   - "dirt"
   - "neon"
   - "mint"
   - "plum"
   - "sunrise"

## Updating Navigation

To add or modify navigation links:

1. Open `_data/navigation.yml`
2. Edit the `main` section:
   ```yaml
   main:
     - title: "New Section"
       url: /new-section/
   ```
3. Create a corresponding page in the `_pages` directory

## Best Practices

- Use clear, descriptive titles for all content
- Include relevant tags and categories for better organization
- Use high-quality images (optimize for web when possible)
- Include excerpts for all posts and tutorials
- Maintain consistent formatting across similar content
- Test all links and embedded content before publishing
