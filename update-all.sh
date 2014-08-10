#!/bin/sh

now="$(date): update all"

git add -A README.md
git add -A CMakeLists.txt
git add -A swig_destina_common.i
git add -A update-all.sh
git add -A src/*
git add -A include/*
git add -A resources/*
git add -A Python/*

git commit -m "$now"

git push origin master
