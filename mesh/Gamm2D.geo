
//commentaires

lc=.05;
lc2=.05;

//lc=.09;
//lc2=.08;

XL = -1;
XR = 2;
YL = 0;
YR = 1;

Point(1) = {XL,YL,0,lc2};
Point(2) = {0,0,0,lc2};
Point(3) = {0.5,-1.1,0,lc2};
Point(4) = {1,0,0,lc2};
Point(5) = {XR,YL,0,lc2};
Point(6) = {XR,YR,0,lc};
Point(7) = {XL,YR,0,lc};

Line(1) = {1,2};
Circle(2) = {2,3,4};
//Circle(3) = {3,4};
Line(3) = {4,5};
Line(4) = {5,6};
Line(5) = {6,7};
Line(6) = {7,1};

Line Loop(1) = {1,2,3,4,5,6};


Plane Surface(1) = {1};

Physical Line("1") = {6};
Physical Line("2") = {4};
Physical Line("4") = {5};
Physical Line("3") = {1,2,3};

Physical Surface("1") = {1};

