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

**Change, add or delete something in repository**
If you save/delete/edit or add some thing in your local machine and you want to
push it to Github. The steps are
1. See the status
2. Add the changes
3. Commit with notes/comments
4. push
```
git status
git add .
git commit -m "note"
git push
```
It may as to pull before push. Then do
```
git pull
```


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
git config --global --edit
```
