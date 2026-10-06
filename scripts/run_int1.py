"""Assemble the fixed INT1 candidate, export, then qualify real new requests."""
from pathlib import Path
import argparse,os,sys,shutil,traceback
from vdc.io import read_json,write_json
from vdc.integration_build import assemble
from vdc.integration_bundle import export_model,load_manifest
from run_all_modules import exclusive_run
from verify_int1 import verify


def main(a):
    run=a.run.resolve();run.mkdir(parents=True,exist_ok=True)
    with exclusive_run(run):
        c=read_json(a.stage/'config.json');source=Path(c['source']);prior=read_json(source/'config.json')
        os.environ['VDC_QWEN_BASE']=prior['model_path']
        write_json(run/'status.json',{'status':'assembling','stage':str(a.stage),'new_lora':False})
        assemble(a.stage,run/'assembly')
        if not (run/'model/bundle.json').exists():export_model(a.stage,run/'model',run/'assembly')
        else:load_manifest(run/'model')
        # Portability is tested by a separate directory; it contains parameters, not validation outputs.
        moved=run/'portable_model'
        if not moved.exists():shutil.copytree(run/'model',moved)
        load_manifest(moved)
        write_json(run/'status.json',{'status':'qualifying','full_gpu_executed':False,'model':str(run/'model')})
        attempt=1
        while (run/f'qualification_{attempt}').exists():attempt+=1
        result=verify(moved,a.stage/'data',run/f'qualification_{attempt}',pk1=prior['pk1_run'])
        write_json(run/'status.json',{'status':result['status'],'full_gpu_executed':result['full_gpu_executed'],
            'model':str(run/'model'),'qualification':str(run/f'qualification_{attempt}'),'new_lora':False,'reserved_queries':False})
        lines=['# VDC INT1 完整应用整合','',f"实际状态：{result['status']}。实际在线Qwen生成：{result['full_gpu_executed']}。",
            '','模型包 model/；新请求与原始Qwen轨迹见 qualification 目录。语义图由同一实际领域adapter的固定文本向量构建，参与参考参数及其依赖读出的离线拟合。',
            '','|视图|固定语义条件|参考范围内单位平均MSE|匹配Ridge MSE|执行|','|---|---|---:|---:|---|']
        for row in read_json(run/'assembly/comparisons.json'):
            lines.append(f"|{row['view']}|{row['condition']}|{row['primary']['macro_unit_mse']}|{row['matched_ridge']['macro_unit_mse']}|{'校验复用' if row['condition']=='zero' else '本轮拟合'}|")
        lines+=['','领域候选预先固定，不按这些结果重新选模型。Qwen/语义参与不等于有科学增益。',
                '','|新请求|执行状态|','|---|---|']
        lines += [f"|{c['case']}|{c['status']}|" for c in result['cases']]
        lines+=['','独立CLI/HTTP与重启、单条/批次一致性、目标隔离和输入敏感性记录在 acceptance.json。',
                '','本次只使用已查看的开发材料，不构成新的独立科学验证；没有新LoRA、预留查询或功能depth。partial保留实际子调用错误，不追加训练网格。',
                '','在线引用检查仅校验本次结果字段和证据编号，机制解释仍待专家审阅。公共响应限原物种和测量协议，不能迁移为killifish功能深度。']
        (run/'REPORT_CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        return 0 if result['status']=='full_verified' else 2

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',type=Path,required=True);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    try:sys.exit(main(a))
    except Exception as exc:
        traceback.print_exc();write_json(a.run/'status.json',{'status':'failed','reason':str(exc),'type':type(exc).__name__});sys.exit(1)
