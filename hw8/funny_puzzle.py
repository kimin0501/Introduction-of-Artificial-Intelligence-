import heapq as hq

def get_manhattan_distance(from_state, to_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: implement this function. This function will not be tested directly by the grader. 

    INPUT: 
        Two states (if second state is omitted then it is assumed that it is the goal state)

    RETURNS:
        A scalar that is the sum of Manhattan distances for all tiles.
    """
    
    # initialize distance
    distance = 0
    
    # iterate the tile index of the from_state array
    for i in range(9):
        # check if there is a blank tile
        if from_state[i] != 0:  
            
            # Compute the current tile's row and column position
            current_row = i // 3
            current_col = i % 3
            
            # Compute the target tile's row and column position
            target_index = to_state.index(from_state[i])
            target_row = target_index // 3
            target_col = target_index % 3
            
            # Calculate the sum of Manhattan distance for all tiles
            distance += abs(current_row - target_row) + abs(current_col - target_col)
    
    return distance

def print_succ(state):
    """
    TODO: This is based on get_succ function below, so should implement that function.

    INPUT: 
        A state (list of length 9)

    WHAT IT DOES:
        Prints the list of all the valid successors in the puzzle. 
    """
    succ_states = get_succ(state)

    for succ_state in succ_states:
        print(succ_state, "h={}".format(get_manhattan_distance(succ_state)))


def get_succ(state):
    """
    TODO: implement this function.

    INPUT: 
        A state (list of length 9)

    RETURNS:
        A list of all the valid successors in the puzzle (don't forget to sort the result as done below). 
    """
    # initializes the list of succ_states and the blank state
    succ_states = []
    blank_states = []
    
    index = 0
    
    # traverse the state to find the empty element (tile with 0)
    for tile in state:
        if tile == 0:
            blank_states.append(index)
        index += 1
    
    # iterate over the every index of blank state 
    for b_index in blank_states:
        # compute the row and column of the empty element
        row = b_index // 3
        col = b_index % 3
        
        # check and swap right
        if col + 1 <= 2 and state[b_index + 1] != 0:  
            tmp = state.copy()
            tmp[b_index] = tmp[b_index + 1]
            tmp[b_index + 1] = 0
            # add the new state into the succ_states array
            succ_states.append(tmp)
        
        # check and swap left
        if col - 1 >= 0 and state[b_index - 1] != 0:  
            tmp = state.copy()
            tmp[b_index] = tmp[b_index - 1]
            tmp[b_index - 1] = 0
            # add the new state into the succ_states array
            succ_states.append(tmp)
        
        # check and swap upwards   
        if row - 1 >= 0 and state[b_index - 3] != 0:  
            tmp = state.copy()
            tmp[b_index] = tmp[b_index - 3]
            tmp[b_index - 3] = 0
            # add the new state into the succ_states array
            succ_states.append(tmp)
        
        # check and swap downwards    
        if row + 1 <= 2 and state[b_index + 3] != 0:  
            tmp = state.copy()
            tmp[b_index] = tmp[b_index + 3]
            tmp[b_index + 3] = 0
            # add the new state into the succ_states array
            succ_states.append(tmp)
    
    
    return sorted(succ_states)




def solve(state, goal_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: Implement the A* algorithm here.

    INPUT:
        An initial state (list of length 9)

    WHAT IT SHOULD DO:
        Prints a path of configurations from initial state to goal state along  h values, number of moves, and max queue number in the format specified in the pdf.
    """
    
    # initialize the variables of the item
    current_state = state.copy()
    g = 0 
    h = get_manhattan_distance(current_state, goal_state) 
    f = g + h 
    parent_index = -1
    item = (f, current_state, (g, h, parent_index))
    
    # initialze the open queue and closed list
    open_list = []
    closed_list = []
    
    # initialize the dictionary to track the index and state
    tracker = dict()
    tracker[0] = item
    
    # add an item to heap queue and current state to closed list
    hq.heappush(open_list, item)
    closed_list.append(current_state)
    
    # initialize max queue to track the maximum length of the queue
    max_length = 1
    
    while open_list:
        # pop the item of the lowest f value
        item = hq.heappop(open_list)
        
        # get the elements of the new item
        current_state = item[1]
        parent_index = closed_list.index(current_state)
        g = item[2][0] + 1

        # check if the current state equals the goal state
        if current_state == goal_state:
            
            # initialize the path to print
            output = [item]
            # track the parent index
            parent_index = item[2][2]
            
            # loop for backtracking
            while parent_index != -1:
                # track the item and index of the parent node
                item = tracker[parent_index]
                parent_index = item[2][2]
                # add the item to the path
                output.append(item)
                
            # # reverse the path
            output.reverse()
            
            # traverse the output to print out the path
            for item in output:
                # initialize variables to print out
                path_print = item[1]
                h_print = item[2][1]
                moves_print = item[2][0]
                # print the output variables of each path
                print(f'{path_print} h={h_print} moves: {moves_print}')
            
            # print the max queue
            print(f'Max queue length: {max_length}')
            return
        
        # initialize the possible successors of the current state using the get_succ method
        successors = get_succ(current_state)
        
        # traverse the successors using update_state helper method
        for successor in successors:
            update_state(successor, g, parent_index, goal_state, open_list, closed_list, tracker)
                
        # update the maximum length 
        if len(open_list) >= max_length:
            max_length = len(open_list)

# helper method to update state            
def update_state(successor, g, parent_index, goal_state, open_list, closed_list, tracker):
    
     # update the elements of the item
    h = get_manhattan_distance(successor, goal_state)
    f = g + h
    new_item = (f, successor, (g, h, parent_index))
    
    # check if the successor is in the closed list (already visited)
    if successor in closed_list:
        # get the index of the visited successor and its item
        visited_index = closed_list.index(successor)
        existing_item = tracker[visited_index]
        prev_g = existing_item[2][0]
        
        # check if the previous g is greater than the current one
        if prev_g > g:
            tracker[visited_index] = new_item
            hq.heappush(open_list, new_item)
            
    # when the successor is new state   
    else:
        # add an item to heap queue and current state to closed list
        hq.heappush(open_list, new_item)
        closed_list.append(successor)
        tracker[len(closed_list) - 1] = new_item

if __name__ == "__main__":
    """
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    """
    # print_succ([2,5,1,4,0,6,7,0,3])
    # print()

    # print(get_manhattan_distance([2,5,1,4,0,6,7,0,3], [1, 2, 3, 4, 5, 6, 7, 0, 0]))
    # print()

    # solve([2,5,1,4,0,6,7,0,3])
    # print()
    
    # print_succ([3, 4, 6, 0, 0, 1, 7, 2, 5])
    # print_succ([6, 0, 0, 3, 5, 1, 7, 2, 4])
    # print_succ([0, 4, 7, 1, 3, 0, 6, 2, 5])
    # solve([3, 4, 6, 0, 0, 1, 7, 2, 5])
    # solve([6, 0, 0, 3, 5, 1, 7, 2, 4])
    solve([0, 4, 7, 1, 3, 0, 6, 2, 5])