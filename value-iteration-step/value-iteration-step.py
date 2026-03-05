def value_iteration_step(values, transitions, rewards, gamma):
    """
    Perform one step of value iteration and return updated values.
    """
    
    num_states = len(values)
    num_actions = len(transitions[0])
    
    new_values = []
    
    for s in range(num_states):
        best = float('-inf')
        
        for a in range(num_actions):
            q = rewards[s][a]
            
            for s_next in range(num_states):
                q += gamma * transitions[s][a][s_next] * values[s_next]
            
            best = max(best, q)
        
        new_values.append(best)
    
    return new_values