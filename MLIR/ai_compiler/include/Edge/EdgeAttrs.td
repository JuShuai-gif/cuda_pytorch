//===- EdgeAttrs.td - Edge dialect 自定义属性与枚举 --------*- tablegen -*-===//
//
// 自定义属性: QuantParamsAttr (量化参数), Layout 枚举 (数据布局).
// 工业动机:
//   - 量化参数 (scale/zeroPoint) 需要挂在算子上, 指导后端选择 INT8 kernel.
//   - 数据布局 (NCHW/NHWC) 直接决定卷积 kernel 的访存效率: TensorRT 会自动
//     在 NCHW/NHWC/NC4HW4 等格式间转换以匹配 Tensor Core, 这里用 Layout 枚举
//     在图层显式表达, 供布局优化 pass 使用.
//
//===----------------------------------------------------------------------===//

#ifndef EDGE_ATTRS
#define EDGE_ATTRS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "Edge/EdgeDialect.td"

// Edge 属性基类
class Edge_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Edge_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// 数据布局枚举
//===----------------------------------------------------------------------===//
def Edge_LayoutNCHW : I32EnumAttrCase<"NCHW", 0, "nchw">;
def Edge_LayoutNHWC : I32EnumAttrCase<"NHWC", 1, "nhwc">;

def Edge_Layout : I32EnumAttr<"Layout", "张量数据布局",
    [Edge_LayoutNCHW, Edge_LayoutNHWC]> {
  let cppNamespace = "::mlir::edge";
}

//===----------------------------------------------------------------------===//
// 量化参数属性
//===----------------------------------------------------------------------===//
def Edge_QuantParamsAttr : Edge_Attr<"QuantParams", "quant_params"> {
  let summary = "量化参数: scale 与 zeroPoint";
  let parameters = (ins "::mlir::FloatAttr":$scale, "int64_t":$zeroPoint);
  let assemblyFormat = "`<` $scale `,` $zeroPoint `>`";
}

#endif // EDGE_ATTRS
