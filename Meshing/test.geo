//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {1, 0, 0, 1.0};
//+
Point(3) = {0, 1, 0, 1.0};
//+
Point(4) = {1, 1, 0, 1.0};

//+
Line(1) = {1, 2};
//+
Line(2) = {4, 2};
//+
Line(3) = {4, 3};
//+
Line(4) = {1, 3};
//+
Curve Loop(1) = {4, -3, 2, -1};
//+
Plane Surface(1) = {1};
//+
Physical Surface(5) = {1};
//+
Physical Curve(6) = {4, 2, 1, 3};
