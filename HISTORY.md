# PlanScore EG Map Development Prompts

This document reconstructs the original prompt sequences for the four different SVG visualization concepts in chronological order.

---

## Graph 1: Grid-based Seat Visualization (render-graph.py → us-states.svg)

**Initial concept**: Create a US state map with grids showing congressional seats

### Prompts in sequence:

1. **Initial setup**
   [commit 689a187](https://github.com/PlanScore/National-EG-Map/commit/689a187f8fed488442765f3f2b5cbdbd475de2a1)
   - Make the new script part of a Makefile, having it accept a single filename argument with a path to a *.pickle file that will store the resulting graph, and put it in a Make rule to generate the file us-states.pickle. Test it out then commit if okay.
   - Add a new script render-graph.py that accepts the pickled graph as CLI input and renders it to an SVG file. It should generate a file called us-states.svg. The output should show state polygons in white with 1px dark grey borders on a transparent background. When trying out the file, open it in Safari so I can preview it.

2. **Add centroids**
   [commit 63c89c9](https://github.com/PlanScore/National-EG-Map/commit/63c89c90ac973ef842f8a4e57da9d330a3cce8f6)
   - Add two new properties to each node in the graph: x and y floats representing the center of the node's geometry in projected coordinates. Verify that they exist in graph creation output, then update the rendering script to place each state's ID at that point in the output SVG.

3. **Add seat counts**
   [commit 447985e](https://github.com/PlanScore/National-EG-Map/commit/447985ec32dc4cdfc14a7da0988b25086af6a83d)
   - Add a seats property to each node of the graph, which contains the number of US Congressional seats for that state from the current 2020 redistricting cycle. Add that number to the rendered SVG output e.g. HI (2) for Hawaii.

4. **Force-directed label positioning**
   [commit 8657496](https://github.com/PlanScore/National-EG-Map/commit/86574961fcead353f0a10213d097d234be698bdb)
   - When rendering, approximate the bounding box for each state label and use a force-directed graph algorithm to space them out. Try to keep labels close to their centroids and expect the Western US to not change much, but in the tighter spacing of the Eastern US we should see the labels move around a bit to make room.

5. **Fix overlaps**
   [commit 3708d89](https://github.com/PlanScore/National-EG-Map/commit/3708d8966337e7b22dd7f397ff16747c5ffe0a64)
   - fix force-directed graph results look bad, let's fix this by making sure the label bboxes do not overlap. Try to keep labels close to their centroids and expect the Western US to not change much, but in the tighter spacing of the Eastern US we should see the labels move around a bit to make room. Right now labels are ending up in weird places for example WI and IL moving much too far west. There should be a force pulling labels toward their centroids, and a hard requirement that they not overlap, and relatively little movement per iteration to let them ease into position. Introduce a requirement that no bboxes overlap, and change the fixed number of iterations to instead stop as soon as no overlaps are detected. Visualize the bboxes as orange rectangles so we can verify that it does the right thing.

6. **Label formatting**
   [commit 0bd0fa7](https://github.com/PlanScore/National-EG-Map/commit/0bd0fa73ff49dcf590e17981f579636d69a970ab)
   - Let's update the labels and bboxes: use a monospace font, cut the font size in half, ensure that the visualized bboxes correctly track the font size changes, and widen them by 10%
   - Make them another 50% wider
   - Make them 20% taller, remove their orange outlines, and use an orange fill instead with 'multiply' mode so overlaps show up
   - Let's now skip labels for any node with no seats

7. **Optimize file size**
   [commit 518c309](https://github.com/PlanScore/National-EG-Map/commit/518c309b37c5be063cf81769ba1afe0b9e025b68)
   - Let's raise the simplification factor to 1000 map units (1km)

8. **Multiline labels**
   [commit 81a133c](https://github.com/PlanScore/National-EG-Map/commit/81a133cb76ecf5738e5e3c3acb95edc8768c9d92)
   - I made some changes and now the labels can be multiline, update calculate_label_bbox() to account for this
   - The orange boxes are supposed to show the label bbox, but they're wide like in the old version. Why different? Fix.
   - Good. The label bboxes look short by 20% fix that
   - Make them 10% taller
   - And 10% narrower

9. **Refactor with dataclass**
   [commit e34bda2](https://github.com/PlanScore/National-EG-Map/commit/e34bda2f46f1f8a365c751ff62dea610608d1cf1)
   - We need a new dataclass with a 'state' and 'seats' property that we can treat as an object in render_graph() and later extend with new features. It should include details of the text label, the orange bbox, and the calculate_label_bbox() logic so that we have one place to instantiate and implement these.

10. **Add seat grid visualization**
    [commit df30762](https://github.com/PlanScore/National-EG-Map/commit/df3076281fc4f38f10b73b2eb4cb504db88c66f1)
    - Let's add a new feature to these labels: we want to show a grid of seat dots for each state, fitting inside the next-largest square e.g. CA's 52 seats fit inside an 8x8 square as a little grid. Let's start with a 4px squares separated by 1px margins, and let's be sure that the overall dimensions cover these new grids so they all lay out nicely.

11. **Fix grid dimensions**
    [commit c5b24a1](https://github.com/PlanScore/National-EG-Map/commit/c5b24a161ca4a56b47c0d319747f12962382bfec)
    - The width needs to accommodate the text width
    - The overall bbox is too tall in all cases

12. **Format tweaks**
    [commit 10de702](https://github.com/PlanScore/National-EG-Map/commit/10de702f85677f3e6fcca0cab8cd839498068474)
    - remove the seats number from the text so it's just the state, then add approx. 4px of margin to the label bbox so they spread out further

13. **Grid spacing improvements**
    [commit 5118587](https://github.com/PlanScore/National-EG-Map/commit/5118587add5044a8ad3f991e913311be6c0d5de8)
    - Let's move the grids down slightly so they stop overlapping the text, then let's double the sizes of the squares, margins, and buffers making up the grids
    - [subsequent requests to fix bbox width issues - specific prompts not preserved in commit message]

14. **Grid aspect ratios**
    [commit 6a3d584](https://github.com/PlanScore/National-EG-Map/commit/6a3d584d567425e254caa1f63af5578554cc5227)
    - CA, IN, and IL should be more vertical than square, while KY, TN, and NC should be more horizontal than square

15. **Alaska and Hawaii transformations**
    [commit 0bcd49f](https://github.com/PlanScore/National-EG-Map/commit/0bcd49fcb8b6c7bf2bb83a41125b1e1636e5ca83)
    - [Multiple prompts about repositioning Alaska and Hawaii - specific sequence not preserved]
    - Commit what we've done with Alaska, Hawaii, and Puerto Rico after linting and formatting

16. **Import standardization**
    [commit f57a07d](https://github.com/PlanScore/National-EG-Map/commit/f57a07d24b9f05dce09bfcab934f60ea3b39c7da)
    - Fix all imports in two ways: don't rename things and don't extract single members from modules
    - [subsequent refinements]

17. **Simplify transformations**
    [commit b849cbc](https://github.com/PlanScore/National-EG-Map/commit/b849cbc3e7dc49fafae8a53336f4dca2211b2e98)
    - Use shapely to simplify translate_geometry, rotate_geometry, and scale_geometry functions

18. **Improve centroid calculation**
    [commit 610a08c](https://github.com/PlanScore/National-EG-Map/commit/610a08c9ffe0137c8fc2bc7b4cd6bb15854c9bec)
    - Move the Hawaii and Alaska changes into two new state-specific functions for clarity
    - When you Calculate centroid from simplified geometry, use the centroid of the largest polygon in a multipolygon instead of the whole-geom centroid

19. **Add efficiency gap coloring**
    [commit dbd6132](https://github.com/PlanScore/National-EG-Map/commit/dbd61323e5e67c91cba9e12a90846704d1fc0d8c)
    - Looks at 4037c00 render-graph2.py for the efficiency gap calculation, and let's apply its result to the grid of squared in render-graphy.py. Copy over load_2024_vote_data() and then apply colors from c853975 render-graph3.py including the white label backgrounds. Show me.

20. **Configurable output filename**
    [commit 7c1d674](https://github.com/PlanScore/National-EG-Map/commit/7c1d674cd780ba5e3616277d26f6ceeb2dc7a241)
    - Wait first make the output filename a required argument to each of the render scripts and in the Makefile
    - All three render scripts

---

## Graph 2: Arrow Visualization (render-graph2.py → us-states2.svg)

**Initial concept**: Replace seat grids with circles/arrows showing efficiency gap

### Prompts in sequence:

1. **Clone and add circles**
   [commit 96506cb](https://github.com/PlanScore/National-EG-Map/commit/96506cb0b1f1e67bda057efb30423b5f3c0f5dc6)
   - Let's clone render-graph.py to render-graph2.py, us-states.svg to us-state2.svg, and add a Make rule to reflect this. We will be working with render-graph2.py moving forward. Let's update the state labels by reading data for the 2024 cycle from 'PlanScore Production Data (2025) - USH Outcomes (2025).tsv', and drawing simple red and blue circles in place of the current grids. The circles should have areas proportional to the number of congressional seats, and be colored red if votes_dem_est < votes_rep_est, blue otherwise. Don't commit, but do run it and show me the result.
   - The circles are ~30% too big
   - Lint, format, and commit

2. **Add efficiency gap**
   [commit 4037c00](https://github.com/PlanScore/National-EG-Map/commit/4037c00f5149425340ae5403fe63541e62816406)
   - Now let's calculate a new way to color the circle using an algorithm called the 'efficiency gap'. When the value is zero the circles should be light gray, then shades of red or blue ranging from -20%/+20%. Use the votes_dem_est and votes_rep_est columns for per-district vote counts, and apply the algorithm from this description: [efficiency gap algorithm description]

3. **Update colors**
   [commit a835c99](https://github.com/PlanScore/National-EG-Map/commit/a835c99e090c2dba44a3ef43c4e4145a5a23d54b)
   - Let's update the colors: bluest should be rgb(0, 73, 168) at 60% opacity, reddest should be rgb(199, 28, 54) at 60% opacity
   - Let's make those orange bboxes white
   - The circles are overlapping the text a little, they should move down by 8px or so

4. **Transform to arrows**
   [commit b78cd2b](https://github.com/PlanScore/National-EG-Map/commit/b78cd2bfb6f6bc02aa54426e39944c51a3e27e39)
   - Let's now change these circles to arrows that point northeast when the EG is republican and northwest when it's democratic
   - Make the arrows twice as large
   - Make the arrows shorter or longer to represent the absolute magnitude of the efficincy gap
   - Make the length adjustment more pronounced
   - Make the longer ones longer

5. **Change angle**
   [commit 3403bde](https://github.com/PlanScore/National-EG-Map/commit/3403bdeeefc2be6f0a2fab7bf39ff22f7129f277)
   - Make them point 30 degrees from horizontal not 45

6. **Configurable output filename** (same as graph 1)
   [commit 7c1d674](https://github.com/PlanScore/National-EG-Map/commit/7c1d674cd780ba5e3616277d26f6ceeb2dc7a241)
   - Wait first make the output filename a required argument to each of the render scripts and in the Makefile
   - All three render scripts

---

## Graph 3: Scaled State Polygons (render-graph3.py → us-states3.svg)

**Initial concept**: Scale state polygon sizes to represent various metrics

### Prompts in sequence:

1. **Clone and scale by representation**
   [commit db8c308](https://github.com/PlanScore/National-EG-Map/commit/db8c3088cc888e98189f874adef3ca541327bd56)
   - let's clone render-graph2.py and its SVG to render-graph3.py, and try a new thing where the state sizes themselves are scaled down from 100% based on area and seat count. Show me, no commits yet

2. **Color states and smaller arrows**
   [commit 37b86c5](https://github.com/PlanScore/National-EG-Map/commit/37b86c57721b719a170e139fc26075137073133d)
   - Okay now color the state the same as the arrow, and make the arrows smaller

3. **Scale by efficiency gap**
   [commit fb7e8c5](https://github.com/PlanScore/National-EG-Map/commit/fb7e8c5aeb4f87a52f13e9ef8d90d2a21ba91a20)
   - Now let's make the area of each state represent the absolute magnitude of the efficiency gap, with 20% as the max. show me.

4. **Scale by seat bias**
   [commit ad49275](https://github.com/PlanScore/National-EG-Map/commit/ad49275fe5ce44b1edb14edce3549161b1f8f810)
   - Okay now let's try a different scaling metric: state size corresponding to bias in SEATS (percentage point EG multiplied by number of seats in the state). Make sure it's the area not just the scale that's affected then show me

5. **Remove arrows**
   [commit c853975](https://github.com/PlanScore/National-EG-Map/commit/c8539757f600a0236de29c73b07dce14b2d79a08)
   - Let's get rid of those arrows, show me
   - Now the label white backgrounds are too short, make them tall enough to cover the text and show it
   - taller by 60%
   - Go taller by 60%, they are still too short
   - taller by 15%
   - taller by 10%

6. **Configurable year**
   [commit 7c1d674](https://github.com/PlanScore/National-EG-Map/commit/7c1d674cd780ba5e3616277d26f6ceeb2dc7a241)
   - Let's make render-graph3.py have a configurable election year. Right now load_2024_vote_data() just looks at one year, but let's make it a configurable first argument to the script and keep 2024 as the default in the Makefile. Show me 2022.
   - Wait first make the output filename a required argument to each of the render scripts and in the Makefile
   - All three render scripts

---

## Graph 4: Transformable SVG Groups (render-graph4.py → us-states4.svg)

**Initial concept**: Create SVG with individually transformable state groups

### Prompts in sequence:

1. **Initial implementation**
   [commit c6fd141](https://github.com/PlanScore/National-EG-Map/commit/c6fd141aea8109e540662bc0dacb8077f87ed17a)
   - [Initial prompt to create render-graph4.py with transformable SVG groups - exact prompt text not preserved in commit message, but commit describes: "Creates SVG output where each state is in a transformable <g> element centered on its largest polygon's centroid. Labels are placed outside groups so they don't scale with transformations." Output matches /tmp/uschart.svg dimensions: 720x400 with Lucida Grande font]

2. **Add offshore boxes**
   [commit 743457a](https://github.com/PlanScore/National-EG-Map/commit/743457a3f319c1c3ba27a8858148093afc8c4d5a)
   - [Prompt to add offshore boxes for tiny northeastern states - exact text not in commit, but resulted in implementing offshore box rendering for 8 tiny northeastern states (VT, NH, MA, RI, CT, NJ, DE, MD) with white rectangles with black borders positioned vertically on the right side of the map]

3. **Test buffered centroids**
   [commit 1e1e7e4](https://github.com/PlanScore/National-EG-Map/commit/1e1e7e408538c55bf76d0cb279459f3a05cbddff)
   - [Prompt to visualize buffered geometry with -10000 buffer for centroid tuning - showing red outlines and markers to help assess buffer impact on centroid placement]

4. **Apply buffer to centroids**
   [commit 55caf28](https://github.com/PlanScore/National-EG-Map/commit/55caf28de1c41e7abb4d8b7e8cfdacf70556b901)
   - [Prompt to use -40000 buffer for centroid calculation to improve label positioning, especially for states like Florida]

5. **Organize SVG structure**
   [commit 5baa3b0](https://github.com/PlanScore/National-EG-Map/commit/5baa3b0bfcea8b7fabd7a5f7eb1dd578b163ecd8)
   - [Prompt to organize classes, IDs, and styles in SVG attributes - exact text not preserved]

6. **Backport changes**
   [commit 0957494](https://github.com/PlanScore/National-EG-Map/commit/095749495b09db7413550e36f04a9da8b8a6cb64)
   - [Backported changes from other branches - exact prompts not preserved]

7. **Add area attribute**
   [commit 7f0bb20](https://github.com/PlanScore/National-EG-Map/commit/7f0bb20b893e9b47e04b37862f6f8670b884e588)
   - Calculate and store the area of each state geometry (excluding offshore boxes) as an integer in a data-area custom SVG attribute on each state shape element. This will be used later to resize elements.

8. **More backports**
   [commit 1da7e52](https://github.com/PlanScore/National-EG-Map/commit/1da7e52cda0cbd034b7a45041e81302c363e5ed7)
   - [Additional backported changes - exact prompts not preserved]

9. **Default to white fills**
   [commit ffe61bd](https://github.com/PlanScore/National-EG-Map/commit/ffe61bd13d1d0044e84e59f451f84866c3b6e689)
   - [Prompt to default to white fills - exact text not preserved]

10. **Remove DC label**
    [commit 0f356d3](https://github.com/PlanScore/National-EG-Map/commit/0f356d395df8106ed5f13de10427ba108b65ca17)
    - [Prompt to remove DC label - exact text not preserved]

---

## Notes

- Graph 1 (us-states.svg) evolved from basic state boundaries to a sophisticated grid-based visualization with force-directed label positioning and efficiency gap coloring
- Graph 2 (us-states2.svg) experimented with circles that evolved into directional arrows showing both magnitude and direction of partisan advantage
- Graph 3 (us-states3.svg) explored scaling state polygon sizes to represent different metrics (representation ratio, efficiency gap magnitude, and seat bias)
- Graph 4 (us-states4.svg) created a more technically sophisticated SVG structure with transformable groups and offshore boxes, though some original prompt text was not preserved in commit messages

The prompts show an iterative, exploratory design process with frequent visual verification and incremental refinements. Where original prompt text was not preserved in git commit messages, the commit descriptions and results are noted in brackets.
