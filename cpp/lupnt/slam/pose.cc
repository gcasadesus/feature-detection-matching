#include "lupnt/slam/pose.h"

namespace lupnt {

  Mat4 MakeTransform(const Mat3& R, const Vec3& t) {
    Mat4 T;
    T.block<3, 3>(0, 0) = R;
    T.block<3, 1>(0, 3) = t;
    T(3, 3) = 1.0;
    return T;
  }

  Mat4 InvertTransform(const Mat4& T) {
    Mat3 R = T.block<3, 3>(0, 0);
    Vec3 t = T.block<3, 1>(0, 3);
    return MakeTransform(R.transpose(), -R.transpose() * t);
  }
}  // namespace lupnt
