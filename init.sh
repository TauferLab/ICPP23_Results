#!/bin/bash

read -p "Experiment Branch Name: " BRANCH

git branch ${BRANCH}
git checkout ${BRANCH}
