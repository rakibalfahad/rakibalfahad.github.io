---
title: "Resources & Learning"
layout: single
permalink: /resources/
author_profile: true
toc: true
toc_sticky: true
toc_label: "Table of Contents"
toc_icon: "cog"
classes: wide
header:
  image: "/images/Gemini_Generated_Image_4dfzcb4dfzcb4dfz.png"
---

Welcome to my comprehensive resources page! Here you'll find tutorials, code snippets, recommended books, courses, and other learning materials to help you on your data science and machine learning journey.

## Tutorials

Check out my step-by-step tutorials on various topics:

{% assign tutorials = site.tutorials %}
{% if tutorials.size > 0 %}
<div class="grid__wrapper">
  {% for post in site.tutorials %}
    {% include archive-single.html type="grid" %}
  {% endfor %}
</div>
{% else %}
<p>Stay tuned for upcoming tutorials!</p>
{% endif %}

## Code Snippets & Tools

Useful code snippets and tools for your projects:

{% assign code_posts = site.code %}
{% if code_posts.size > 0 %}
<div class="grid__wrapper">
  {% for post in site.code %}
    {% include archive-single.html type="grid" %}
  {% endfor %}
</div>
{% else %}
<p>Check back soon for useful code snippets and tools!</p>
{% endif %}

## Recommended Books

* [The Elements of Statistical Learning](https://web.stanford.edu/~hastie/ElemStatLearn/) by Hastie, Tibshirani, and Friedman
* [Deep Learning](https://www.deeplearningbook.org/) by Goodfellow, Bengio, and Courville
* [Python for Data Analysis](https://wesmckinney.com/book/) by Wes McKinney
* [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) by Aurélien Géron
* [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) by Christopher M. Bishop

## Online Courses

* [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning) by Andrew Ng on Coursera
* [Machine Learning](https://www.coursera.org/learn/machine-learning) by Andrew Ng on Coursera
* [Fast.ai Practical Deep Learning for Coders](https://course.fast.ai/)
* [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/) from Stanford
* [DataCamp Courses](https://www.datacamp.com/)
* [edX Data Science & AI Programs](https://www.edx.org/)

## Useful Websites & Blogs

* [Towards Data Science](https://towardsdatascience.com/)
* [Papers With Code](https://paperswithcode.com/)
* [Distill.pub](https://distill.pub/)
* [PyImageSearch](https://pyimagesearch.com/)
* [Machine Learning Mastery](https://machinelearningmastery.com/)
* [KDnuggets](https://www.kdnuggets.com/)
* [Analytics Vidhya](https://www.analyticsvidhya.com/)
* [AI Summer](https://theaisummer.com/)

## Tools & Libraries

* [TensorFlow](https://www.tensorflow.org/) - Deep learning framework
* [PyTorch](https://pytorch.org/) - Deep learning framework
* [scikit-learn](https://scikit-learn.org/) - Machine learning library
* [Pandas](https://pandas.pydata.org/) - Data manipulation and analysis
* [NumPy](https://numpy.org/) - Numerical computing
* [Matplotlib](https://matplotlib.org/) - Data visualization
* [Seaborn](https://seaborn.pydata.org/) - Statistical data visualization
* [Hugging Face](https://huggingface.co/) - NLP and transformer models
* [NLTK](https://www.nltk.org/) - Natural language processing
* [spaCy](https://spacy.io/) - Industrial-strength NLP
