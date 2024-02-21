import numpy as np
class HiddenMarkovModel:
    """
    Class for Hidden Markov Model 
    """

    def __init__(self, observation_states: np.ndarray, hidden_states: np.ndarray, prior_p: np.ndarray, transition_p: np.ndarray, emission_p: np.ndarray):
        """

        Initialization of HMM object

        Args:
            observation_states (np.ndarray): observed states 
            hidden_states (np.ndarray): hidden states 
            prior_p (np.ndarray): prior probabities of hidden states 
            transition_p (np.ndarray): transition probabilites between hidden states
            emission_p (np.ndarray): emission probabilites from transition to hidden states 
        """             
        
        self.observation_states = observation_states
        self.observation_states_dict = {state: index for index, state in enumerate(list(self.observation_states))}

        self.hidden_states = hidden_states
        self.hidden_states_dict = {index: state for index, state in enumerate(list(self.hidden_states))}
        
        self.prior_p= prior_p
        self.transition_p = transition_p
        self.emission_p = emission_p

        #raise error if input array is empty
        if len(observation_states) <= 0 or len(hidden_states) <= 0 or len(prior_p) <= 0 or len(transition_p) <= 0 or len(emission_p) <= 0:
            raise ValueError("input must be a not empty ndarray.")
        
        #raise error if input prior p is not sum to 1
        if not np.isclose(np.sum(prior_p), 1):
            raise ValueError("The sum of prior_p must be close to 1.")



    def forward(self, input_observation_states: np.ndarray) -> float:
        """
        TODO 

        This function runs the forward algorithm on an input sequence of observation states

        Args:
            input_observation_states (np.ndarray): observation sequence to run forward algorithm on 

        Returns:
            forward_probability (float): forward probability (likelihood) for the input observed sequence  
        """        

        # Step 1. Initialize variables
        hidden_states = self.hidden_states
        prior_p = self.prior_p
        transition_p = self.transition_p
        emission_p = self.emission_p
        # observation_state_sequence = self.observation_states
        observation_indices =self.observation_states_dict
        # Initialize the forward matrix
        observation_state_sequence = input_observation_states
        observation_states_dict = []
        for obs in observation_state_sequence:
            observation_states_dict.append(observation_indices[obs])

        # Step 2. Calculate probabilities
        # Initialize the forward matrix
        F = np.zeros((len(observation_state_sequence), len(hidden_states)))
        F[0, :] = prior_p * emission_p[:, observation_states_dict[0]]
        # Iteration step
        for t in range(1, len(observation_state_sequence)):
            for s in range(len(hidden_states)):
                F[t, s] = np.sum(F[t-1, :] * transition_p[:, s]) * emission_p[s, observation_states_dict[t]]

        # Step 3. Return final probability 
        forward_probability = np.sum(F[-1, :])
        print("Forward Probability of the sequence is:", forward_probability)

        return forward_probability


    def viterbi(self, decode_observation_states: np.ndarray) -> list:
        """
        TODO

        This function runs the viterbi algorithm on an input sequence of observation states

        Args:
            decode_observation_states (np.ndarray): observation state sequence to decode 

        Returns:
            best_hidden_state_sequence(list): most likely list of hidden states that generated the sequence observed states
        """        
        
        # Step 1. Initialize variables
        
        #store probabilities of hidden state at each step 

        #store best path for traceback
        best_path = np.zeros(len(decode_observation_states))   

        hidden_states = self.hidden_states
        prior_p = self.prior_p
        transition_p = self.transition_p
        emission_p = self.emission_p
        # observation_state_sequence = self.observation_states
        observation_indices =self.observation_states_dict
        observation_state_sequence = decode_observation_states
        observation_states_dict = []
        for obs in observation_state_sequence:
            observation_states_dict.append(observation_indices[obs])
        
        print(hidden_states)
        viterbi_table = np.zeros((len(hidden_states), len(observation_state_sequence)))
        path = np.zeros((len(hidden_states), len(observation_state_sequence)), dtype=int)

        
       
       # Step 2. Calculate Probabilities
        for s in range(len(hidden_states)):
            viterbi_table[s, 0] = prior_p[s] * emission_p[s, observation_states_dict[0]]
            path[s, 0] = 0
            
        # Step 3. Traceback 
        for t in range(1, len(observation_state_sequence)):
            new_path = np.zeros((len(hidden_states), len(observation_state_sequence)), dtype=int)
            for s in range(len(hidden_states)):
                (prob, state) = max((viterbi_table[s_prime, t-1] * transition_p[s_prime, s] * emission_p[s, observation_states_dict[t]], s_prime) for s_prime in range(len(hidden_states)))
                viterbi_table[s, t] = prob
                new_path[s, :t+1] = path[state, :t+1]
                new_path[s, t] = s
            path = new_path

        # Step 4. Return best hidden state sequence 
        best_prob = max(viterbi_table[:, -1])
        best_path_index = np.argmax(viterbi_table[:, -1])
        best_path = path[best_path_index, :]

        best_hidden_state_sequence = [hidden_states[state] for state in best_path]

        print("Best Hidden State Sequence:", best_hidden_state_sequence)

        return best_hidden_state_sequence

        