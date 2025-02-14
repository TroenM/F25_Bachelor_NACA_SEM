//+
nodeSize = 1.0;

//+
Point(1) = {0, 0, 0, nodeSize};
//+
Point(2) = {0.5, 0, 0, nodeSize};
//+
Point(3) = {1, 0, 0, nodeSize};
//+
Point(4) = {0, 0.5, 0, nodeSize};
//+
Point(5) = {1, 0.5, 0, nodeSize};
//+
Point(6) = {0, 1, 0, nodeSize};
//+
Point(7) = {0.5, 1, 0, nodeSize};
//+
Point(8) = {1, 1, 0, nodeSize};
//+


//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3,5};
//+
Line(4) = {5,8};
//+
Line(5) = {8,7};
//+
Line(6) = {7,6};
//+
Line(7) = {6,4};
//+
Line(8) = {4,1};
//+
Curve Loop(1) = {1, 2, 3, 4, 5, 6, 7,8};
//+
Plane Surface(1) = {1};
//+
Physical Curve("Inlet", 9) = {8, 7};
//+
Physical Curve("Outlet", 10) = {3, 4};
//+
Physical Curve("Top", 11) = {5,6};
//+
Physical Curve("Bottom", 12) = {1, 2};
//+
Physical Surface("Fluid", 13) = {1};

//+
Transfinite Surface {1} = {1, 3, 8, 6};
//+
Transfinite Curve {7, 4, 8} = 3 Using Progression 1;
//+
Transfinite Curve {4, 3} = 3 Using Progression 1;
//+
Transfinite Curve {5, 6} = 3 Using Progression 1;
//+
Transfinite Curve {1, 2} = 3 Using Progression 1;
//+
Transfinite Curve {7, 8} = 2 Using Progression 1;
//+
Transfinite Curve {4, 3} = 2 Using Progression 1;
//+
Transfinite Curve {5, 6} = 2 Using Progression 1;
//+
Transfinite Curve {1, 2} = 2 Using Progression 1;
//+
Transfinite Curve {7, 8} = 3 Using Progression 1;
//+
Transfinite Curve {4, 3} = 3 Using Progression 1;
//+
Transfinite Curve {7, 8} = 2 Using Progression 1;
//+
Transfinite Curve {4, 3} = 2 Using Progression 1;
//+
Transfinite Curve {6, 5} = 2 Using Progression 1;
//+
Transfinite Curve {1, 2} = 2 Using Progression 1;
//+
Transfinite Curve {7, 8} = 3 Using Progression 1;
//+
Transfinite Curve {4, 3} = 3 Using Progression 1;
//+
Transfinite Curve {7, 8} = 1 Using Progression 1;
//+
Transfinite Curve {3, 4} = 1 Using Progression 1;
//+
Transfinite Curve {6, 5} = 1 Using Progression 1;
//+
Transfinite Curve {1, 2} = 1 Using Progression 1;
