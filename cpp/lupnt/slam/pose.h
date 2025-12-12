#pragma once

#include "lupnt/core/definitions.h"
namespace lupnt {

  Mat4 MakeTransform(const Mat3& R, const Vec3& t);
  Mat4 InvertTransform(const Mat4& T);

  const Mat4d FLU_T_RDF = Mat4d{{0.0, 0.0, 1.0, 0.0},
                                {-1.0, 0.0, 0.0, 0.0},
                                {0.0, -1.0, 0.0, 0.0},
                                {0.0, 0.0, 0.0, 1.0}};
  const Mat4d RDF_T_FLU = InvertTransform(FLU_T_RDF);

  const Mat4d FLU_T_OCV = FLU_T_RDF;
  const Mat4d OCV_T_FLU = RDF_T_FLU;
}  // namespace lupnt
