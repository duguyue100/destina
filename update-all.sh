#!/bin/sh

now="$(date): update all"

git add README.md
git add CMakeLists.txt
git add update-all.sh
git add src/*
git add include/*
git add resources/*

git commit -m "$now"

git push origin master
