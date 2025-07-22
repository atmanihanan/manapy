// Gmsh project created on Wed Dec 18 00:49:47 2024
SetFactory("OpenCASCADE");
//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {0, 1, 0, 1.0};
//+
Point(3) = {1, 0, 0, 1.0};
//+
Point(4) = {1, 1, 0, 1.0};
//+
Line(1) = {2, 1};
//+
Line(2) = {1, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 2};
//+
Curve Loop(1) = {1, 2, 3, 4};
//+
Plane Surface(1) = {1};
//+
Physical Curve("in", 5) = {1};
//+
Physical Curve("out", 6) = {3};
//+
Physical Curve("upper", 7) = {4};
//+
Physical Curve("bottom", 8) = {2};
//+
Physical Surface("fluid", 9) += {1};
