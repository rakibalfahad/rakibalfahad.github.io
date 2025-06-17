---
title: "Resources & Learning"
layout: single
permalink: /resources/
author_profile: true
classes: wide
header:
  image: "/images/Gemini_Generated_Image_4dfzcb4dfzcb4dfz.png"
---

# Resources & Learning Materials

Welcome to my resources page! Here you'll find learning materials, code samples, and helpful resources for data science and machine learning.

## Learning Resources

### Tutorials
{% if site.tutorials.size > 0 %}
<ul>
  {% for tutorial in site.tutorials %}
    <li><a href="{{ tutorial.url }}">{{ tutorial.title }}</a> - {{ tutorial.excerpt | strip_html | truncate: 100 }}</li>
  {% endfor %}
</ul>
{% else %}
<p>Coming soon! Check back for tutorials on data science, machine learning, and programming.</p>
{% endif %}

### Code Snippets
{% if site.code.size > 0 %}
<ul>
  {% for code in site.code %}
    <li><a href="{{ code.url }}">{{ code.title }}</a> - {{ code.excerpt | strip_html | truncate: 100 }}</li>
  {% endfor %}
</ul>
{% else %}
<p>Coming soon! Check back for useful code snippets and examples.</p>
{% endif %}

## Books I Recommend

<div class="resource-section">
  <ul>
    <li><a href="https://web.stanford.edu/~hastie/ElemStatLearn/">The Elements of Statistical Learning</a> by Hastie, Tibshirani, and Friedman</li>
    <li><a href="https://www.deeplearningbook.org/">Deep Learning</a> by Goodfellow, Bengio, and Courville</li>
    <li><a href="https://wesmckinney.com/book/">Python for Data Analysis</a> by Wes McKinney</li>
    <li><a href="https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/">Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow</a> by Aurélien Géron</li>
    <li><a href="https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/">Pattern Recognition and Machine Learning</a> by Christopher M. Bishop</li>
  </ul>
</div>

## Online Courses

<div class="resource-section">
  <ul>
    <li><a href="https://www.coursera.org/specializations/deep-learning">Deep Learning Specialization</a> by Andrew Ng on Coursera</li>
    <li><a href="https://www.coursera.org/learn/machine-learning">Machine Learning</a> by Andrew Ng on Coursera</li>
    <li><a href="https://course.fast.ai/">Fast.ai Practical Deep Learning for Coders</a></li>
    <li><a href="http://cs231n.stanford.edu/">CS231n: Convolutional Neural Networks for Visual Recognition</a> from Stanford</li>
  </ul>
</div>

## Helpful Websites

<div class="resource-section">
  <ul>
    <li><a href="https://towardsdatascience.com/">Towards Data Science</a> - Articles on data science, machine learning, and AI</li>
    <li><a href="https://paperswithcode.com/">Papers With Code</a> - Latest research papers with code implementations</li>
    <li><a href="https://distill.pub/">Distill.pub</a> - Clear explanations of machine learning concepts</li>
    <li><a href="https://www.kdnuggets.com/">KDnuggets</a> - News, tutorials, and opinions on data science</li>
  </ul>
</div>

## Essential Tools & Libraries

<div class="resource-section">
  <div class="tools-row">
    <div class="tools-column">
      <h4>Machine Learning & Deep Learning</h4>
      <ul>
        <li><a href="https://www.tensorflow.org/">TensorFlow</a></li>
        <li><a href="https://pytorch.org/">PyTorch</a></li>
        <li><a href="https://scikit-learn.org/">scikit-learn</a></li>
      </ul>
    </div>
    
    <div class="tools-column">
      <h4>Data Processing</h4>
      <ul>
        <li><a href="https://pandas.pydata.org/">Pandas</a></li>
        <li><a href="https://numpy.org/">NumPy</a></li>
        <li><a href="https://dask.org/">Dask</a></li>
      </ul>
    </div>
    
    <div class="tools-column">
      <h4>Visualization</h4>
      <ul>
        <li><a href="https://matplotlib.org/">Matplotlib</a></li>
        <li><a href="https://seaborn.pydata.org/">Seaborn</a></li>
        <li><a href="https://plotly.com/">Plotly</a></li>
      </ul>
    </div>
  </div>
</div>
