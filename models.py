from abc import ABC, abstractmethod
from collections import deque
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

class ARCModel(ABC):
    @abstractmethod
    def train(self, train_examples):
        """Train the model on given training examples"""
        pass

    @abstractmethod
    def predict(self, input_data):
        """Predict output for given input data"""
        pass


class DummyModel(ARCModel):
    def train(self, train_examples):
        self.train_examples = train_examples

    def predict(self, input_data):
        # Simple dummy prediction
        rows = len(input_data)
        cols = len(input_data[0]) if rows > 0 else 0
        return [[0 for _ in range(cols)] for _ in range(rows)]

class Object:
    """A helper class to store info about a found object."""
    def __init__(self, pixels, grid):
        self.pixels = pixels # List of (r, c) tuples
        self.pixel_count = len(pixels)

        # Find bounds
        min_r = min(r for r, c in pixels)
        max_r = max(r for r, c in pixels)
        min_c = min(c for r, c in pixels)
        max_c = max(c for r, c in pixels)

        self.position = (min_r, min_c)
        self.height = max_r - min_r + 1
        self.width = max_c - min_c + 1

        self.shape = [[0 for _ in range(self.width)] for _ in range(self.height)]
        self.colors = set()

        for r, c in pixels:
            color = grid[r][c]
            # Handle pixels that are part of the object but have background color (0)
            # This happens in "hollow" shapes.
            if color != 0:
                self.colors.add(color)
            self.shape[r - min_r][c - min_c] = color

    def __repr__(self):
        return f"Object(pos={self.position}, size={self.height}x{self.width}, colors={self.colors})"

    def __eq__(self, other):
        # Helper for comparing objects, useful for some tasks
        if not isinstance(other, Object):
            return False
        return self.shape == other.shape and self.colors == other.colors

    def __hash__(self):
        # Make objects hashable
        return hash((tuple(tuple(row) for row in self.shape), frozenset(self.colors)))


class ProgramSynthesisModel(ARCModel):
    """
    Upgraded model that uses Breadth-First Search (BFS) to
    find a *sequence* of transformations.
    """

    def __init__(self, max_depth=2, verbose=False):
        """
        Args:
            max_depth (int): Max program steps to search (e.g., 2 = "do A, then do B")
            verbose (bool): Print when a solution is found
        """
        super().__init__()
        self.solution_program = None
        self.max_depth = max_depth
        self.verbose = verbose
        self.all_colors = list(range(1, 10)) # Colors 1-9

        # This is our library of "verbs"
        self.transformation_library = self._build_transformation_library()

    def _hash_grid(self, grid):
        """Converts a grid (list of lists) to a hashable tuple."""
        return tuple(tuple(row) for row in grid)

    # --- Core Object-Finding Logic (Unchanged) ---
    def _find_objects(self, grid, include_background=False):
        # (This function is identical to the previous version)
        height = len(grid)
        width = len(grid[0]) if height > 0 else 0
        visited = [[False for _ in range(width)] for _ in range(height)]
        objects = []
        background_color = 0 if not include_background else -1
        for r in range(height):
            for c in range(width):
                color = grid[r][c]
                if color != background_color and not visited[r][c]:
                    q = deque([(r, c)])
                    visited[r][c] = True
                    object_pixels = []
                    object_color = color if not include_background else -1
                    while q:
                        curr_r, curr_c = q.popleft()
                        object_pixels.append((curr_r, curr_c))
                        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            nr, nc = curr_r + dr, curr_c + dc
                            if 0 <= nr < height and 0 <= nc < width and \
                               not visited[nr][nc]:
                                if include_background:
                                    if grid[nr][nc] == background_color:
                                        visited[nr][nc] = True
                                        q.append((nr, nc))
                                else:
                                    # For objects, find same-colored neighbors
                                    # (Let's change this to find *contiguous* objects,
                                    # not just same-colored)
                                    if grid[nr][nc] != 0: # <-- Key change
                                        visited[nr][nc] = True
                                        q.append((nr, nc))
                    if object_pixels:
                        objects.append(Object(object_pixels, grid))
        return objects

    # --- Filter Library (Unchanged) ---
    def _filter_by_color(self, objects, color):
        return [obj for obj in objects if color in obj.colors]
    def _filter_largest(self, objects):
        if not objects: return []
        max_size = max(obj.pixel_count for obj in objects)
        return [obj for obj in objects if obj.pixel_count == max_size]
    def _filter_smallest(self, objects):
        if not objects: return []
        min_size = min(obj.pixel_count for obj in objects)
        return [obj for obj in objects if obj.pixel_count == min_size]
    def _filter_all(self, objects):
        return objects

    # --- Action Library (Unchanged) ---
    def _action_recolor(self, grid, objects, new_color):
        if not objects: return grid
        new_grid = [row[:] for row in grid]
        for obj in objects:
            for r, c in obj.pixels:
                if new_grid[r][c] != 0:
                    new_grid[r][c] = new_color
        return new_grid
    def _action_delete(self, grid, objects):
        if not objects: return grid
        new_grid = [row[:] for row in grid]
        for obj in objects:
            for r, c in obj.pixels:
                new_grid[r][c] = 0
        return new_grid

    # --- Simple Geometric Actions (Unchanged) ---
    def _action_flip_x(self, grid): return np.flip(np.array(grid), axis=0).tolist()
    def _action_flip_y(self, grid): return np.flip(np.array(grid), axis=1).tolist()
    def _action_rotate_90(self, grid): return np.rot90(np.array(grid), 1).tolist()
    def _action_rotate_180(self, grid): return np.rot90(np.array(grid), 2).tolist()
    def _action_rotate_270(self, grid): return np.rot90(np.array(grid), 3).tolist()

    # --- NEW: Transformation Library Builder ---

    def _build_transformation_library(self):
        """Creates the full list of callable, 1-step transformations."""
        library = {} # Use a dict to give names {name: function}

        # 1. Add Geometric Transformations
        library['flip_x'] = lambda g: self._action_flip_x(g)
        library['flip_y'] = lambda g: self._action_flip_y(g)
        library['rotate_90'] = lambda g: self._action_rotate_90(g)
        library['rotate_180'] = lambda g: self._action_rotate_180(g)
        library['rotate_270'] = lambda g: self._action_rotate_270(g)

        # 2. Build Object-Based Transformations

        # Define filters {name: function}
        filters = {
            'all_obj': self._filter_all,
            'largest_obj': self._filter_largest,
            'smallest_obj': self._filter_smallest,
        }
        for c in self.all_colors:
            filters[f'color_{c}_obj'] = lambda objs, color=c: self._filter_by_color(objs, color)

        # Define actions {name: function}
        actions = {
            'delete': self._action_delete
        }
        for c in self.all_colors:
            actions[f'recolor_to_{c}'] = lambda grid, objs, color=c: self._action_recolor(grid, objs, color)

        # Combine filters and actions into single transformations
        for f_name, f_func in filters.items():
            for a_name, a_func in actions.items():

                # Use a helper to create the function to avoid lambda closure issues
                def _create_transform(filter_func, action_func):
                    def transform(grid):
                        objects = self._find_objects(grid)
                        filtered_objects = filter_func(objects)
                        return action_func(grid, filtered_objects)
                    return transform

                name = f'{a_name}({f_name})'
                library[name] = _create_transform(f_func, a_func)

        return library

    # --- NEW: BFS Solver (The `train` method) ---

    def train(self, train_examples):
        """
        Finds a program (sequence of transformations) using BFS.
        """
        self.solution_program = None
        if not train_examples:
            return

        # We find the program using the *first* example,
        # then verify it with all others.
        start_grid = train_examples[0]['input']
        target_grid = train_examples[0]['output']

        # Queue stores (grid_state, program_so_far)
        # Program is a list of transformation names
        queue = deque([(start_grid, [])])

        # Visited set stores hashable grids to avoid loops
        visited = {self._hash_grid(start_grid)}

        while queue:
            current_grid, program_names = queue.popleft()

            # --- Check for Solution ---
            if current_grid == target_grid:
                # Found a potential program!
                # Now, verify it against ALL training examples.
                if self._check_program_against_all_examples(train_examples, program_names):
                    if self.verbose:
                        print(f"Found solution: {program_names}")
                    # Build the final, callable function
                    self.solution_program = self._build_program_from_list(program_names)
                    return # SUCCESS!

            # --- Limit Search Depth ---
            if len(program_names) >= self.max_depth:
                continue # Stop searching this branch

            # --- Expand Search (Apply all transformations) ---
            for name, transform_func in self.transformation_library.items():

                new_grid = transform_func(current_grid)
                new_grid_hash = self._hash_grid(new_grid)

                if new_grid_hash not in visited:
                    visited.add(new_grid_hash)
                    new_program = program_names + [name]
                    queue.append((new_grid, new_program))

    # --- NEW: Solver Helper Functions ---

    def _check_program_against_all_examples(self, train_examples, program_names):
        """Checks if a program works for all training pairs."""
        for example in train_examples:
            grid = example['input']
            target = example['output']

            # Apply each transformation in the program
            for name in program_names:
                grid = self.transformation_library[name](grid)

            if grid != target:
                return False # Program failed on this example
        return True # Program worked for all examples

    def _build_program_from_list(self, program_names):
        """Converts a list of names into a single callable function."""
        def program(grid):
            for name in program_names:
                grid = self.transformation_library[name](grid)
            return grid
        return program

    # --- NEW: `predict` method ---

    def predict(self, input_data):
        """
        Apply the composed program found during training.
        """
        if self.solution_program:
            return self.solution_program(input_data)
        else:
            # Fallback: return input
            return [row[:] for row in input_data]