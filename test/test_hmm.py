import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath('/Users/yifeichen/Desktop/winter24/HW6-HMM/'))

import pytest
from hmm import HiddenMarkovModel
import numpy as np




def test_mini_weather():
    """
    TODO: 
    Create an instance of your HMM class using the "small_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "small_weather_input_output.npz" file.

    Ensure that the output of your Forward algorithm is correct. 

    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    In addition, check for at least 2 edge cases using this toy model. 
    """

    mini_hmm=np.load('./data/mini_weather_hmm.npz')
    mini_input=np.load('./data/mini_weather_sequences.npz')

    hmm_ = HiddenMarkovModel(mini_hmm['observation_states'],mini_hmm['hidden_states'], mini_hmm['prior_p'],mini_hmm['transition_p'],mini_hmm['emission_p'])

    forward_result = hmm_.forward(mini_input['observation_state_sequence'])
    viterbi_result = hmm_.viterbi(mini_input['observation_state_sequence'])

    assert forward_result == pytest.approx(0.0350644, abs=1e-6)
    assert len(viterbi_result) == 5
    assert viterbi_result == ['hot', 'cold', 'cold', 'hot', 'cold']

    # two test below are test for the empty input
    empty_input = []
    with pytest.raises(ValueError) as e:
        hmm_ = HiddenMarkovModel(empty_input,mini_hmm['hidden_states'], mini_hmm['prior_p'],mini_hmm['transition_p'],mini_hmm['emission_p'])
    with pytest.raises(ValueError) as e:
        hmm_ = HiddenMarkovModel(mini_hmm['observation_states'],mini_hmm['hidden_states'], empty_input,mini_hmm['transition_p'],mini_hmm['emission_p'])

    #  test below are test for the wrong prior_p value
    not_1_prior_p = np.ones(3)
    with pytest.raises(ValueError) as e:
        hmm_ = HiddenMarkovModel(mini_hmm['observation_states'],mini_hmm['hidden_states'], not_1_prior_p,mini_hmm['transition_p'],mini_hmm['emission_p'])


   



def test_full_weather():

    """
    TODO: 
    Create an instance of your HMM class using the "full_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "full_weather_input_output.npz" file
        
    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    """

    full_hmm=np.load('./data/full_weather_hmm.npz')
    full_input=np.load('./data/full_weather_sequences.npz')

    hmm_ = HiddenMarkovModel(full_hmm['observation_states'],full_hmm['hidden_states'], full_hmm['prior_p'],full_hmm['transition_p'],full_hmm['emission_p'])

    viterbi_result = hmm_.viterbi(full_input['observation_state_sequence'])

    assert len(viterbi_result) == 16
    assert viterbi_result == ['hot', 'temperate', 'temperate', 'temperate', 'temperate', 'temperate', 'cold', 'cold', 'freezing', 'freezing', 'freezing', 'freezing', 'freezing', 'cold', 'cold', 'temperate']













