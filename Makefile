.PHONY: clean

# Default target
us-states.pickle: create-graph.py
	python create-graph.py us-states.pickle

clean:
	rm -f us-states.pickle