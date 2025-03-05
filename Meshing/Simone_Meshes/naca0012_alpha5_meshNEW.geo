// Gmsh project created on Tue Oct 08 13:02:34 2024

Include "naca0012_alpha5.geo";

s= 1.034;

//+
Point(131) = {-7, s, 0, 1.0};
//+
Point(132) = {-7, -2, 0, 1.0};
//+
Point(133) = {13, -2, 0, 1.0};
//+
Point(134) = {13, s, 0, 1.0};


//+
Line(12) = {131, 132};
//+
Line(13) = {132, 133};
//+
Line(14) = {133, 134};
//+
Line(15) = {131, 134};
//+
Curve Loop(1) = {15, -14, -13, -12};
//+
Curve Loop(2) = {1};
//+
Plane Surface(1) = {1, 2};



//+
Physical Curve("in", 1) = {12};
//+
Physical Curve("out", 2) = {14};
//+
Physical Curve("bed", 3) = {13};
//+
Physical Curve("naca", 5) = {1};

//+
Physical Curve("fs", 4) = {15};


//Recombine Surface {1};

//+
Transfinite Curve {1} = 50*6 Using Progression 1;
//+
Transfinite Curve {13} = 40*8 Using Progression 1;
//+
Transfinite Curve {12} = 10*7 Using Progression 1.02;
//+
Transfinite Curve {14} = 10*7 Using Progression 0.99;

//+
Physical Surface("fluid", 154) = {1};
//+
Transfinite Curve {15} = 100*5 Using Progression 1;


Mesh.Algorithm = 12; // Use quadrilateral meshing algorithm

