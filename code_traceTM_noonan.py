# Sophia Noonan Thoery Project 2
# This code implements a simulator for Non-deterministic Turing Machines (NTM)
# It can read machine descriptions from files and simulate their execution and behavior on input strings

import csv
import argparse
from typing import List, Tuple, Dict, Set
from dataclasses import dataclass

@dataclass
class Transition:
    """represents a single transition in the Turing Machine
    """
    current_state: str    
    input_char: str     
    next_state: str     
    write_char: str      
    direction: str     

@dataclass
class Configuration:
    """a complete configuration of the turing machine
    
    attributes:
        left_tape: contents of the tape to the left of the head
        current_state: current state
        head_char: character under the head
        right_tape: contents of the tape to the right of the head
        parent: reference to the previous configuration
        depth: number of steps from the initial 
    """
    left_tape: str       
    current_state: str  
    head_char: str       
    right_tape: str      
    parent: 'Configuration' = None  
    depth: int = 0       

class TuringMachine:
    """implementation of a NTM simulator"""
    
    def __init__(self, filename: str):
        """initialize the machine from a description file

        """
        # basic machine components based on the tuple
        self.name = ""                          
        self.states: Set[str] = set()           
        self.input_alphabet: Set[str] = set()   
        self.tape_alphabet: Set[str] = set()    
        self.start_state: str = ""              
        self.accept_state: str = ""             
        self.reject_state: str = "qreject"      
        self.transitions: Dict[Tuple[str, str], List[Transition]] = {}  # transition rules of the machine
        
        #metrics for analyzing machine behavior
        self.total_configurations = 0           
        self.max_tree_depth = 0                
        self.branching_factors = []            
        
        self._load_from_file(filename)

    def _load_from_file(self, filename: str) -> None:
        """load machine description from a CSV file
        
        my CSVs files contain
        - machine name
        - set of states
        - input alphabet
        - tape alphabet
        - start state
        - accept state
        - reject state
        - transition rules
        """
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            # Read machine components
            self.name = next(reader)[0]
            self.states = set(next(reader)[0].split())
            self.input_alphabet = set(next(reader)[0].split())
            self.tape_alphabet = set(next(reader)[0].split())
            self.start_state = next(reader)[0]
            self.accept_state = next(reader)[0]
            self.reject_state = next(reader)[0]

            #read in the transition rules
            for row in reader:
                if len(row) != 5:
                    continue
                curr_state, input_char, next_state, write_char, direction = row
                key = (curr_state, input_char)
                if key not in self.transitions:
                    self.transitions[key] = []
                self.transitions[key].append(
                    Transition(curr_state, input_char, next_state, write_char, direction)
                )

    def get_next_configurations(self, config: Configuration) -> List[Configuration]:
        """generate all possible next configurations from the current one
        
        for NTMs, there might be multiple possible next configurations.
        if no transition is defined, moves to reject state.
        
        """
        next_configs = []
        key = (config.current_state, config.head_char)
        
        #if no transition exists, reject
        if key not in self.transitions:
            return [Configuration(
                config.left_tape,
                self.reject_state,
                config.head_char,
                config.right_tape,
                parent=config,
                depth=config.depth + 1
            )]

        #apply each possible transition
        for transition in self.transitions[key]:
            new_left = config.left_tape
            new_right = config.right_tape
            
            #handle left movement
            if transition.direction == 'L':
                if new_left:  #if there's tape to the left
                    head_char = new_left[-1]
                    new_left = new_left[:-1]
                    new_right = transition.write_char + new_right 
                else:  #if we're at the left edge
                    head_char = '_' 
                    new_right = transition.write_char + new_right
            else:  #handle right movement
                if new_right:  #if there's tape to the right
                    head_char = new_right[0]
                    new_right = new_right[1:]
                    new_left = new_left + transition.write_char  
                else:  #if we're at the right edge
                    head_char = '_' 
                    new_left = new_left + transition.write_char

            next_configs.append(Configuration(
                new_left,
                transition.next_state,
                head_char,
                new_right,
                parent=config,
                depth=config.depth + 1
            ))

        return next_configs

    def trace_computation(self, input_string: str, max_depth: int = 1000) -> Tuple[str, int, float, List[Configuration], Dict]:
        """simulate the machine's computation on the input string
        
        Performs a breadth-first search of all possible computation paths.
        
        this returns:
            tuple containing:
            - result ("accept", "reject", or "timeout")
            - depth reached
            - degree of nondeterminism
            - path to accepting configuration (if accepted)
            - computation metrics
        """
        #create initial configuration
        initial_config = Configuration(
            "",
            self.start_state,
            input_string[0] if input_string else '_',
            input_string[1:],
            None,
            0
        )

        #initialize BFS levels
        levels = [[initial_config]]
        self.total_configurations = 1
        self.max_tree_depth = 0
        self.branching_factors = []
        
        #explore configurations level by level
        while levels[-1] and len(levels) <= max_depth:
            current_level = []
            
            # calculate branching factor for current level
            if len(levels) > 1:
                branching_factor = len(levels[-1]) / len(levels[-2])
                self.branching_factors.append(branching_factor)
            
            #process each configuration at current level
            for config in levels[-1]:
                #check for acceptance
                if config.current_state == self.accept_state:
                    path = self._get_path(config)
                    metrics = self._get_metrics(levels)
                    return "accept", len(path) - 1, self._calculate_nondeterminism(levels), path, metrics
                
                #generate next configurations if not in reject state
                if config.current_state != self.reject_state:
                    next_configs = self.get_next_configurations(config)
                    current_level.extend(next_configs)
                    self.total_configurations += len(next_configs)
            
            #check for rejection
            if not current_level:
                metrics = self._get_metrics(levels)
                return "reject", len(levels) - 1, self._calculate_nondeterminism(levels), [], metrics
            
            levels.append(current_level)
            self.max_tree_depth = len(levels) - 1
            
        #return timeout if max depth reached
        metrics = self._get_metrics(levels)
        return "timeout", len(levels) - 1, self._calculate_nondeterminism(levels), [], metrics

    def _get_metrics(self, levels: List[List[Configuration]]) -> Dict:
        """calculate computation metrics
        
        returns a dictionaruy of:
        - total configurations explored
        - maximum tree depth
        - average branching factor
        - number of configurations at each level
        """
        return {
            "total_configurations": self.total_configurations,
            "max_tree_depth": self.max_tree_depth,
            "average_branching_factor": sum(self.branching_factors) / len(self.branching_factors) if self.branching_factors else 1.0,
            "configurations_per_level": [len(level) for level in levels]
        }

    def _get_path(self, config: Configuration) -> List[Configuration]:
        """reconstruct the path from initial configuration to current"""
        path = []
        current = config
        while current:
            path.append(current)
            current = current.parent
        return list(reversed(path))

    def _calculate_nondeterminism(self, levels: List[List[Configuration]]) -> float:
        """calculate the average branching factor across all levels"""
        if len(levels) <= 1:
            return 1.0
        
        total_branching = 0
        for i in range(len(levels) - 1):
            if levels[i]: 
                branching = len(levels[i + 1]) / len(levels[i])
                total_branching += branching
                
        return total_branching / (len(levels) - 1)

def format_configuration(config: Configuration) -> str:
    """format a configuration as a string for display"""
    return f"({config.left_tape}, {config.current_state}, {config.head_char}{config.right_tape})"

def main():
    """main function for the command-line interface"""
    #parse command line arguments
    parser = argparse.ArgumentParser(description='NTM Tracer')
    parser.add_argument('machine_file', help='Path to the NTM description file')
    parser.add_argument('input_string', help='Input string to process')
    parser.add_argument('--max-depth', type=int, default=1000, help='Maximum computation depth')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    
    args = parser.parse_args()
    
    #create and run the machine
    tm = TuringMachine(args.machine_file)
    print(f"Machine name: {tm.name}")
    print(f"Input string: {args.input_string}")
    
    result, depth, nondeterminism, path, metrics = tm.trace_computation(args.input_string, args.max_depth)
    
    #display computation statistics to the user 
    print("\nComputation Statistics:")
    print(f"Total configurations explored: {metrics['total_configurations']}")
    print(f"Maximum tree depth reached: {metrics['max_tree_depth']}")
    print(f"Average branching factor: {metrics['average_branching_factor']:.2f}")
    print(f"Nondeterminism degree: {nondeterminism:.2f}")
    
    #show detailed debug information if requested
    if args.debug:
        print("\nConfigurations per level:")
        for level, count in enumerate(metrics['configurations_per_level']):
            print(f"Level {level}: {count} configurations")
    
    # display the final result
    if result == "accept":
        print(f"\nString accepted in {depth} steps")
        print("Execution path:")
        for config in path:
            print(format_configuration(config))
    elif result == "reject":
        print(f"\nString rejected in {depth} steps")
    else:
        print(f"\nExecution stopped after {depth} steps (timeout)")

if __name__ == "__main__":
    main()