// Package demo is a tiny package for corpus tests.
//
// This second paragraph is not indexed.
package demo

// Widget is a thing with a name.
type Widget struct{ Name string }

// NewWidget creates a widget.
func NewWidget(name string) *Widget { return &Widget{Name: name} }

// Rename changes the widget's name.
func (w *Widget) Rename(name string) { w.Name = name }

// Add returns the sum of a and b.
func Add(a, b int) int { return a + b }

// Old does nothing.
//
// Deprecated: use Add.
func Old() {}

// Deprecated: gone.
func Gone() {}

func undocumented() {}

// hidden is unexported.
func hidden() {}
