.PHONY: clean

# Default target
us-states.svg: us-states.pickle render-graph.py
	python render-graph.py us-states.pickle us-states.svg

us-states2.svg: us-states.pickle render-graph2.py
	python render-graph2.py us-states.pickle us-states2.svg

us-states3.svg: us-states.pickle render-graph3.py
	python render-graph3.py us-states.pickle us-states3.svg 2024

us-states3-2022.svg: us-states.pickle render-graph3.py
	python render-graph3.py us-states.pickle us-states3-2022.svg 2022

us-states.pickle: create-graph.py
	python create-graph.py us-states.pickle

clean:
	rm -f us-states.pickle us-states.svg us-states2.svg us-states3.svg us-states3-2022.svg