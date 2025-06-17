---
title: "Resources"
layout: single
permalink: /resources/
author_profile: false
header:
  image: "/images/Gemini_Generated_Image_4dfzcb4dfzcb4dfz.png"
---

<div class="page__content-wrapper">
  <div class="page__content">
    <h1>Resources & Learning Materials</h1>
    
    <p>Welcome to my curated collection of resources for data science, machine learning, and programming.</p>
    
    <section class="resource-block">
      <h2>Learning Materials</h2>
      
      <div class="resource-section">
        <h3>Tutorials</h3>
        {% if site.tutorials.size > 0 %}
        <ul class="resource-list">
          {% for tutorial in site.tutorials %}
            <li><a href="{{ tutorial.url }}">{{ tutorial.title }}</a></li>
          {% endfor %}
        </ul>
        {% else %}
        <p>Coming soon! Check back for tutorials on data science and machine learning.</p>
        {% endif %}
      </div>
      
      <div class="resource-section">
        <h3>Code Examples</h3>
        {% if site.code.size > 0 %}
        <ul class="resource-list">
          {% for code in site.code %}
            <li><a href="{{ code.url }}">{{ code.title }}</a></li>
          {% endfor %}
        </ul>
        {% else %}
        <p>Coming soon! Check back for code snippets and examples.</p>
        {% endif %}
      </div>
    </section>
    
    <section class="resource-block">
      <h2>Books & Courses</h2>
      
      <div class="resource-section">
        <h3>Recommended Books</h3>
        <ul class="resource-list">
          <li><a href="https://web.stanford.edu/~hastie/ElemStatLearn/">The Elements of Statistical Learning</a></li>
          <li><a href="https://www.deeplearningbook.org/">Deep Learning</a> by Goodfellow, Bengio, and Courville</li>
          <li><a href="https://wesmckinney.com/book/">Python for Data Analysis</a> by Wes McKinney</li>
          <li><a href="https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/">Hands-On Machine Learning</a></li>
        </ul>
      </div>
      
      <div class="resource-section">
        <h3>Online Courses</h3>
        <ul class="resource-list">
          <li><a href="https://www.coursera.org/specializations/deep-learning">Deep Learning Specialization</a> (Coursera)</li>
          <li><a href="https://www.coursera.org/learn/machine-learning">Machine Learning</a> by Andrew Ng</li>
          <li><a href="https://course.fast.ai/">Fast.ai Practical Deep Learning</a></li>
          <li><a href="http://cs231n.stanford.edu/">CS231n: CNN for Visual Recognition</a></li>
        </ul>
      </div>
    </section>
    
    <section class="resource-block">
      <h2>Tools & References</h2>
      
      <div class="resource-section">
        <h3>Key Libraries</h3>
        <div class="tools-grid">
          <div class="tools-card">
            <h4>TensorFlow</h4>
            <p>Deep learning framework</p>
            <a href="https://www.tensorflow.org/" class="tools-link">Visit →</a>
          </div>
          
          <div class="tools-card">
            <h4>PyTorch</h4>
            <p>Deep learning library</p>
            <a href="https://pytorch.org/" class="tools-link">Visit →</a>
          </div>
          
          <div class="tools-card">
            <h4>scikit-learn</h4>
            <p>Machine learning toolkit</p>
            <a href="https://scikit-learn.org/" class="tools-link">Visit →</a>
          </div>
          
          <div class="tools-card">
            <h4>Pandas</h4>
            <p>Data manipulation</p>
            <a href="https://pandas.pydata.org/" class="tools-link">Visit →</a>
          </div>
          
          <div class="tools-card">
            <h4>NumPy</h4>
            <p>Numerical computing</p>
            <a href="https://numpy.org/" class="tools-link">Visit →</a>
          </div>
          
          <div class="tools-card">
            <h4>Matplotlib</h4>
            <p>Data visualization</p>
            <a href="https://matplotlib.org/" class="tools-link">Visit →</a>
          </div>
        </div>
      </div>
      
      <div class="resource-section">
        <h3>Useful Websites</h3>
        <ul class="resource-list">
          <li><a href="https://towardsdatascience.com/">Towards Data Science</a></li>
          <li><a href="https://paperswithcode.com/">Papers With Code</a></li>
          <li><a href="https://distill.pub/">Distill.pub</a></li>
          <li><a href="https://www.kdnuggets.com/">KDnuggets</a></li>
        </ul>
      </div>
    </section>
  </div>
</div>
