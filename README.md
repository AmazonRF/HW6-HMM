![BuildStatus](https://github.com/AmazonRF/HW6-HMM/actions/workflows/pytest.yml/badge.svg?event=push)

# Hidden Markov Model (HMM) Project

This project implements a Hidden Markov Model (HMM) in Python, featuring algorithms for calculating observation likelihood using the Forward Algorithm and determining the most likely sequence of hidden states with the Viterbi Algorithm.

## Overview

The HMM is a statistical model that assumes an underlying process generating a sequence of observations is a Markov process with unobservable (hidden) states. This project includes:

- Implementation of the Forward Algorithm to calculate the likelihood of a sequence of observations.
- Implementation of the Viterbi Algorithm to find the most likely sequence of hidden states given a sequence of observations.

## Getting Started

### Prerequisites

- Python 3.x
- NumPy

Ensure you have Python and NumPy installed on your system. If not, you can install NumPy using pip:

```bash
pip install numpy


### Grading 

* Algorithm implementation (6 points)
    * Forward algorithm is correct (2)
    * Viterbi is correct (2)
    * Output is correct on small weather dataset (1)
    * Output is correct on full weather dataset (1)

* Unit Tests (3 points)
    * Mini model unit test (1)
    * Full model unit test (1)
    * Edge cases (1)

* Style (1 point)
    * Readable code and updated README with a description of your methods 

* Extra credit (0.5 points)
    * Pip installable and Github actions (0.5)
