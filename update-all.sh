#!/bin/sh

now="$(date): update all"

git add -A README.md
git add -A CMakeLists.txt
git add -A update-all.sh
git add -A src/*
git add -A include/*
git add -A resources/*

git commit -m "$now"

git push origin master
