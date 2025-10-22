.PHONY: clean

# Default target
us-states.svg: us-states.pickle render-graph.py
	python render-graph.py us-states.pickle

us-states.pickle: create-graph.py
	python create-graph.py us-states.pickle

clean:
	rm -f us-states.pickle us-states.svg