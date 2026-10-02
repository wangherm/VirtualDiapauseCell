# 分模块测试与分步研究发布

## 发布的是已验证能力，不是整个仓库的笼统成功

代码实现、CPU测试、GPU加载、真实数据接入、科学评价和批准使用分别记录。代码测试通过不能自动将science_status改为validated。

建议里程碑（不是本次已上线版本）：

| 里程碑 | 允许启用 | 仍不启用 |
|---|---|---|
| R0 软件alpha | 数据审核、知识/数值接口、合成smoke | 任何实际biology结论 |
| R1 状态研究版 | 指定系统中的programme恢复、经验证clock | 通用depth、跨物种gene生成 |
| R2 波形/扰动研究版 | 有足够数据的wave、KO端点或一步释放预测 | 稀疏样本的精确峰谷、多步模拟 |
| R3 功能研究版 | 指定协议与端点的功能读出 | 无监督通用休眠深度或存活概率 |

可以只批准programme恢复而暂不批准clock，反之亦然；不必等所有能力全过关。失败模块的真实结果保留，服务不返回该能力的随机预测或全零占位答案。

## 各模块的最低测试

- 数据：shape、namespace、原样本去重、独立单位与共享control跨split检查；data_role与time origin一致。
- 状态：mask loss、训练集拟合预处理、identity/简单数值基线、保存重载、断点恢复；改变上游特征顺序即失败。
- Wave：参考只来自获准train；保留原幅度；Flat与Unclassified；超出覆盖范围明确不输出或标外推；基于biotime的相对顺序不是物理速度。
- 响应：endpoint与transition模式分开；没有dt不补0；持续基因型不再施加一次KO；没有额外动作仍允许自然变化；不伪造单细胞配对。
- 功能：endpoint/protocol/单位必须齐全；缺少target即不训练；screen fitness不能当RNA target。
- 知识：真实Qwen加载、tokenizer边界、completion mask、重载、论文家族隔离、正确/打乱/无语义消融。
- 发布：模型hash、scope、评价文件hash和人工科学审查一致；synthetic权重禁止发布成研究模型。

## 科学评价与审查

`evaluate-state`会生成带checkpoint hash的实数据评价记录，但不会自动通过科学门槛。研究者需事先固定评价任务、比较方法、容许误差/效应标准、独立单位及适用范围。没有预先标准时，不从看起来最好的结果临时选阈值。

最低应同时检查programme恢复与真实偏离保留、clock与独立未参与定位的信号、错误与覆盖率，不能只看token loss或只计算被保留的容易样本。当前自带evaluate-state的stress test仅测人为增加损坏前的观测恢复，独立生物学重复/功能实验需另外接入，不能将该stress test单独包装为全面生物验证。

人工review文件示意（这里故意不提供可直接绕过审批的已填批准JSON）：

```json
{
  "decision": "approved_for_declared_research_scope",
  "reviewed_by": "填写真实审核者",
  "checkpoint_sha256": "best.pt的实际SHA256",
  "scope_hash": "run.json中scope的object_hash",
  "evaluation_path": "实际evaluation.json路径",
  "evaluation_sha256": "实际评价文件SHA256",
  "approved_capabilities": ["programme_reconstruction"]
}
```

scope哈希可在Python中用`vdc.io.object_hash(manifest['scope'])`计算。审核文件并不替代统计证据；release检查的是文件关联和明确审批，无法自动判断所有生物学论证。

## 共享核心变动后的回归测试

下游模型依赖特定输入表示。修改encoder、归一化、programme成员、semantics或clock参考时，创建新run，不覆盖已批准版本；重新运行相关依赖模块的回归测试。当前不创建复杂服务依赖系统：通过run/representation hashes和固定工件做到这一点。

回滚就是将调用切回旧的完整release目录；不要把旧decoder、新encoder、新programme词表随机拼起来。发布目录不可覆盖。

## 本地服务

当前`service.py`只服务状态模型，训练和读取原始数据不在HTTP请求中执行。只绑定127.0.0.1，适合本机或SSH tunnel。未实现公网认证、访问控制、任务队列、隔离执行、大文件上传、批量用户并发，不应直接暴露公网。

本轮只用FastAPI TestClient和模拟审批/预测对象验证请求契约，没有真实生物模型API部署。未来响应模块取得科学批准后，在同一服务调用层加端点即可，不复制一套训练逻辑。
