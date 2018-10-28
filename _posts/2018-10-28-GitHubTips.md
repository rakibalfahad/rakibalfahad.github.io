---
title: "GitHub Tips"
date: 2018-10-28
tags: [GitHub]
header:
  #image: "/images/KerasTensorflow.jpg"
excerpt: "GitHub Tips"
mathjax: "true"
---
# Description
In this page I will try to give you some helpful GitHub tips that I have collected
over the time

**Git asks for username every time I push in linux**
ref: [Link](https://stackoverflow.com/questions/11403407/git-asks-for-username-every-time-i-push)

To set the user name and password
```
git config --global credential.https://github.com.username <your_username>
git config --global credential.https://github.com.password <your_password>
```
This works on a site by site basis and modifies your global git config.
To see the changes, use:
```
git config --global --edi
```
