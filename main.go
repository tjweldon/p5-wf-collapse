package main

import (
	"flag"
	"fmt"
	"image/color"
	"log"
	"math"
	"os"
	"runtime/pprof"
	"time"

	"github.com/go-p5/p5"
)

// Program Constants
const (
	// W is the grid width in tiles
	W int = 30

	// H is the grid height in tiles
	H int = 20

	// NodeSize is the pixel size of a tile
	NodeSize = 50
)

// Side is an enum that enumerates the sides of a Tile
type Side int

const (
	North Side = iota
	East
	South
	West

	_SideCount
)

func (s *Side) String() string {
	return [_SideCount]string{"N", "E", "S", "W"}[*s]
}

// Opposite is a convenience function for swapping between neighboring perspectives on which
// Side is being constrained. I.e. two horizontally adjacent tiles see the side on which the
// constraint is applied as East and West i.e. East and East.Opposite().
func (s Side) Opposite() Side {
	return [_SideCount]Side{
		North: South,
		East:  West,
		South: North,
		West:  East,
	}[s]
}

// Offset is a representation of the center to edge offset in pixels for a tile. It is a pair
// of numbers (a vector) that is the distance from the center of the tile to the center of the
// side supplied.
func (s Side) Offset() (x, y float64) {
	l := float64(NodeSize) / 2.0
	switch s {
	case North:
		y = -l
	case East:
		x = l
	case South:
		y = l
	case West:
		x = -l
	}

	return x, y
}

// Pos is a grid position (i.e. a discrete vector)
type Pos struct {
	X, Y int
}

func (p *Pos) String() string {
	return fmt.Sprintf("(%d, %d)", p.X, p.Y)
}

// Equals checks if two Pos instances refer to the same grid position.
func (p Pos) Equals(q Pos) bool {
	return p.X == q.X && p.Y == q.Y
}

// InBounds returns true if the Pos is within the grid
func (p *Pos) InBounds() bool {
	return (p.X >= 0 && p.Y >= 0 && p.X < W && p.Y < H)
}

// Next is an iterator that returns positions that are always in bounds,
// and in order from left to right, moving to the beginning of the next row
// after reaching the end of the current row. At the end of the final row
// it returns nil to signal that it is exhausted.
func (p *Pos) Next() *Pos {
	x := (p.X + 1) % W
	y := p.Y
	if x == 0 {
		y = y + 1
	}

	if y == H {
		return nil
	}
	return &Pos{x, y}
}

// Adj retuns the position that is adjacent to p on the side supplied. If the
// result would be out of bounds, nil is returned
func (p *Pos) Adj(s Side) *Pos {
	var a *Pos
	switch s {
	case North:
		a = &Pos{p.X, p.Y - 1}
	case East:
		a = &Pos{p.X + 1, p.Y}
	case South:
		a = &Pos{p.X, p.Y + 1}
	case West:
		a = &Pos{p.X - 1, p.Y}
	}
	if !a.InBounds() {
		return nil
	}

	return a
}

// Cardinal returns the adjacent Pos for each Side. It always returns a four
// element array, with nils in the positions that would be out of grid bounds
func (p *Pos) Cardinal() [_SideCount]*Pos {
	c := [_SideCount]*Pos{}
	for s := North; s < _SideCount; s++ {
		c[s] = p.Adj(s)
	}

	return c
}

// ScreenPos maps the grid position to a pixel location
func (p *Pos) ScreenPos() (x, y float64) {
	return float64(NodeSize*p.X) + 25.0, float64(NodeSize*p.Y) + 25.0
}

// FromScreenPos initialises a grid position for a given pixel coordinate struct (used for mouse events)
func FromScreenPos(sp struct{ X, Y float64 }) Pos {
	return Pos{
		X: int((sp.X - 25.0) / float64(NodeSize)),
		Y: int((sp.Y - 25.0) / float64(NodeSize)),
	}
}

// Idx is an explicit implementation of the ordering implied by Next that maps directly to an index.
// The mapping is reversible, see FromIndex.
func (p Pos) Idx() int {
	return p.Y*W + p.X
}

// FromIndex maps a positive integer to a Pos, it is the inverse of Pos.Idx. i.e:
//
//		if:		q := FromIndex(p.Idx())
//	 then: 	p.Equals(q)  true
func FromIndex(idx int) Pos {
	return Pos{idx % W, idx / W}
}

// Tile is a specific state, i.e. a particular choice of PATH or NO_PATH for each side.
// this is represented internally as an array of bools.
type Tile struct {
	Sides [_SideCount]bool
}

// Grid is the container for the wavefunction values
type Grid map[Pos]*TileState

// Draw for Grid iterates through all grid positions and calls draw on the wavefunction value at that position
func (g *Grid) Draw() {
	for p, node := range *g {
		node.PutAt(p)
		node.Draw()
	}
}

// MakeGridWithFactory is a grid intialiser that allows the calling code to define the function used to initialise the
// wavefunction value at each position.
func MakeGridWithFactory(factory func(*Pos) *TileState) Grid {
	grid := make(Grid)

	initial := &Pos{}
	// this is the main use case for the Pos.Next iterator
	for p := initial; nil != p; p = p.Next() {
		grid[*p] = factory(p)
	}

	return grid
}

// ========================================================================== //
//                                                                            //
// TileState struct (this is where the bulk of the algorithm is implemented)  //
//                                                                            //
// ========================================================================== //

// TileState is the representation of state superposition for a single tile. The abstraction
// used is a vector of states, i.e. TileState.Vec.
type TileState struct {
	// Vec contains the enumeration of all possible future outcomes for the TileState
	Vec []Tile

	// Pos is the Grid Pos for which thie TileState describes the state
	Pos

	drawConf struct {
		freq   [_SideCount]int
		beta   [_SideCount]float64
		alpha  [_SideCount]uint8
		width  [_SideCount]float64
		cx, cy float64
	}
}

// NewTileState initialises a TileState and uses the fact that the binary representation
// of the numbers 0..15 inclusive contains all possible combinations of four On/Off values
// to populate the state vector with every combination of sides having paths or not.
func NewTileState() *TileState {
	v := []Tile{}
	for i := 0; i < 1<<4; i++ {
		sides := [_SideCount]bool{
			(i>>North)%2 == 0,
			(i>>East)%2 == 0,
			(i>>South)%2 == 0,
			(i>>West)%2 == 0,
		}
		v = append(v, Tile{Sides: sides})
	}

	return &TileState{
		Vec: v,
		Pos: Pos{},
		drawConf: struct {
			freq  [_SideCount]int
			beta  [_SideCount]float64
			alpha [_SideCount]uint8
			width [_SideCount]float64
			cx    float64
			cy    float64
		}{},
	}
}

// Draw draws a single TileState in the position it aught to occupy
func (t *TileState) Draw() {

	// draw the tile border
	p5.StrokeWidth(1)
	p5.Fill(color.Transparent)
	p5.Stroke(color.RGBA{127, 127, 127, 255})
	x, y := t.Pos.ScreenPos()
	p5.Rect(x, y, NodeSize, NodeSize)

	// go through all the states and count how many times each side has a path
	maxFreq := len(t.Vec)

	// iterating over tile sides
	for s, frequency := range t.drawConf.freq {
		if frequency != 1 {
			continue
		}
		// set the path fragment width
		p5.StrokeWidth(t.drawConf.width[s])

		// set the alpha level
		p5.Stroke(color.RGBA{0, 0, 0, t.drawConf.alpha[s]})

		// this sets how long the frament is as a function of how likely it is that
		// the fragment will be poplulated on observation.
		sideChance := float64(frequency) / float64(maxFreq)
		dx, dy := Side(s).Offset()
		dx, dy = sideChance*dx, sideChance*dy

		// draw the path fragment
		p5.Line(t.drawConf.cx, t.drawConf.cy, t.drawConf.cx+dx, t.drawConf.cy+dy)
	}

}

// refreshDrawConf precalculates a bunch of stuff needed for drawing when the tiles are initialised
// and when their states change
func (t *TileState) refreshDrawConf() {
	for side := North; side < _SideCount; side++ {
		freq := len(t.StatesHaving(side, true))
		t.drawConf.freq[side] = freq
		t.drawConf.beta[side] = float64(freq) / 16.0
		t.drawConf.alpha[side] = uint8(50.0*t.drawConf.beta[side] + 255.0*(1.0-t.drawConf.beta[side]))
		t.drawConf.width[side] = float64(NodeSize)*t.drawConf.beta[side] + 2.0*(1.0-t.drawConf.beta[side])
	}
}

// PutAt is the TileState.Pos setter
func (t *TileState) PutAt(p Pos) {
	t.Pos = p
	t.refreshDrawConf()
	x, y := p.ScreenPos()
	t.drawConf.cx, t.drawConf.cy = x+float64(NodeSize)/2.0, y+float64(NodeSize)/2.0
}

// StatesOf returns an enumeration of possible outcomes for a particular side. If it could
// go either way the result is []bool{false, true}, otherwise it's either of:
//
//   - []bool{false} if the TileState has no outcomes where the passed side has a path, or
//   - []bool{true} if the TileState only has outcomes where the pased side has a path.
//
// If it returns []bool{} then something has gone horribly wrong.
func (t *TileState) StatesOf(side Side) []bool {
	states := []bool{t.Vec[0].Sides[side]}
	if len(t.Vec) == 1 {
		return states
	}
	for _, tile := range t.Vec[1:] {
		if tile.Sides[side] != states[0] {
			states = append(states, tile.Sides[side])
			break
		}
	}

	return states
}

// Entropy is a simplified expression of Shannon Entropy, making use of the fact that all
// outcomes are equally likely to simplify the maths.
func (t *TileState) Entropy() float64 {
	degeneracy := float64(len(t.Vec))
	sChance := 1.0 / degeneracy
	return -math.Log(sChance)
}

func (t *TileState) IsCollapsed() bool {
	return len(t.Vec) == 1
}

// Observe is the action that causes a TileState to collapse to a single outcome. This is propagated
// as far as it needs to be to maintain adjacency constraints.
func (t *TileState) Observe(wf *WaveFunction) {
	// select a state index at random
	choice := int(p5.Random(0.0, float64(len(t.Vec))))

	// set the Vec of outocomes to only contain the choice
	t.Vec = []Tile{t.Vec[choice]}
	t.refreshDrawConf()

	// create an empty array to hold references to visited positions
	visited := [W * H]bool{}

	// propagate the new information into the system, culling any incompatible outcomes.
	t.Propagate(wf, visited[:])
}

// Propagate recurses through the grid, visiting each TileState at most once. It iterates through neighboring TileStates
// and calls Constrain to ensure that each of them has culled any states that are not compatible with this TileState.
// Propagate stops recursing when either all TileStates have been visited or when, after applying constraints to its neighboring
// TileStates it finds that no states were culled (i.e. propagation of the collapse has provided as much information
// about the system as it can until another observation is made).
func (t *TileState) Propagate(wf *WaveFunction, visited []bool) {
	// if this node has been visited, we're done
	isVisited := func(ts *TileState) bool {
		return visited[ts.Pos.Idx()]
	}
	if isVisited(t) {
		return
	}

	// add current to visited
	visited[t.Pos.Idx()] = true

	var lowestEntropyNeighbour *TileState
	statesCulled := 0
	// iterate over neighboring positions
	for _, pos := range t.Cardinal() {
		// out of bounds positions appear as nil
		if pos == nil {
			continue
		}

		// check that the grid has something at that position and that it hasn't been visited yet
		neighbour := wf.StateAt(*pos)
		if neighbour == nil {
			continue
		}
		if isVisited(neighbour) {
			continue
		}

		// constrain the neighbour's states and keep track of how many outcomes were precluded by
		// the application of the constraints
		statesCulled += neighbour.Constrain(wf)

		// keep track of which neighbour has the lowest Entropy
		if lowestEntropyNeighbour == nil {
			lowestEntropyNeighbour = neighbour
		} else if lowestEntropyNeighbour.Entropy() >= neighbour.Entropy() {
			lowestEntropyNeighbour = neighbour
		}

	}

	// If either no new nodes were visited or the application of constraints didn't tell us anything
	// more about the eventual state of the system then don't continue to propagate
	if lowestEntropyNeighbour == nil || statesCulled == 0 {
		return
	}

	// if we could still remove possible outcomes from the system, then continue to propagate
	lowestEntropyNeighbour.Propagate(wf, visited)
}

// Constrain checks the neighboring TileStates and removes any states from this instance if they
// are incompatible with any of the states that a neighbour could resolve to on observation.
// Constrain returns the number of states that were removed from the possible outcomes for the TileState.
func (t *TileState) Constrain(wf *WaveFunction) int {
	initial := len(t.Vec)
	for s, pos := range t.Pos.Cardinal() {
		if pos == nil {
			continue
		}

		neighbour := wf.StateAt(*pos)
		side := Side(s)
		if neighbour == nil {
			continue
		}

		// for the shared side, which outcomes are possible according to the neighbour
		neighbourStates := neighbour.StatesOf(side.Opposite())

		// there should never be 0 possible outcomes, so in the case that there is two
		// then this doesn't constrain our state at all so we continue to the next
		// neighbour
		if len(neighbourStates) != 1 {
			continue
		}

		// otherwise we grab the one outcome that the neighbour can have and filter our own
		// states down to only those compatible with our neighbour
		t.Vec = t.StatesHaving(side, neighbourStates[0])
	}
	// refresh drawing params
	culled := initial - len(t.Vec)
	if culled > 0 {
		t.refreshDrawConf()
	}

	// return the number of states culled by the application of the constraint
	return culled
}

// StatesHaving returns the subset if this TileState's outcomes as a []Tile which have the sideState outcome for the supplied side.
// For example the subset of states that have a path going north would be tileState.StatesHaving(North, true).
func (t *TileState) StatesHaving(side Side, sideState bool) []Tile {
	states := []Tile{}
	for _, tile := range t.Vec {
		if tile.Sides[side] == sideState {
			states = append(states, tile)
		}
	}

	return states
}

type WaveFunction struct {
	States [W * H]*TileState
}

func (wf *WaveFunction) StateAt(p Pos) *TileState {
	return wf.States[p.Idx()]
}

func FromGrid(g Grid) *WaveFunction {
	wfStates := [W * H]*TileState{}
	for idx := range wfStates {
		pos := FromIndex(idx)
		tileState, exists := g[pos]
		if exists {
			tileState.PutAt(FromIndex(idx))
			wfStates[idx] = tileState
		}
	}

	return &WaveFunction{States: wfStates}
}

func (wf *WaveFunction) Draw() {
	for _, ts := range wf.States {
		ts.Draw()
	}
}

func (wf *WaveFunction) GetNext() *TileState {
	var next *TileState
	for _, ts := range wf.States {
		if ts == nil {
			continue
		}
		if ts.IsCollapsed() {
			continue
		}
		if next == nil {
			next = ts
		} else if ts.Entropy() < next.Entropy() {
			next = ts
		}
	}
	return next
}

// Python port only needs to go up to here, can ignore all other functions from here down
func IterOneFrom(p Pos, wf *WaveFunction) *TileState {
	if !p.InBounds() {
		return nil
	}

	ts := wf.StateAt(p)
	if ts == nil {
		return nil
	}

	if len(ts.Vec) == 0 {
		return nil
	}

	ts.Observe(wf)
	return wf.GetNext()
}

func RunWorker(p Pos, wf *WaveFunction) <-chan *TileState {
	work := func(pos Pos, wavFunc *WaveFunction, out chan<- *TileState) {
		defer close(out)
		out <- IterOneFrom(pos, wavFunc)
	}

	c := make(chan *TileState)
	go work(p, wf, c)

	return c
}

var cpuprofile = flag.String("cpuprofile", "", "write cpu profile to file")

func main() {
	flag.Parse()
	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			log.Fatal(err)
		}
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}
	// set up the model
	grid := MakeGridWithFactory(
		func(p *Pos) *TileState {
			return NewTileState()
		},
	)
	var p *Pos
	p = &Pos{}
	waveFunction := FromGrid(grid)

	setup := func() {
		// standard p5 boilerplate bs
		p5.Canvas(NodeSize*W+50, NodeSize*H+50)
		p5.Background(color.White)
	}

	draw := func() {
		// mouse handling for interactive collapse, implments Observe/Collapse on click
		// recursively draw the model
		waveFunction.Draw()
	}

	// run the p5 process
	go p5.Run(setup, draw)

	for next := waveFunction.StateAt(*p); next != nil; next = IterOneFrom(*p, waveFunction) {
		time.Sleep(time.Second / 60)
		p = &next.Pos
	}
}
