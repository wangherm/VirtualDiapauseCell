"""Private source audit and admission. This command never starts training."""
from pathlib import Path
import argparse
import collections
import html
import json
import os
from vdc.io import read_json, object_hash
from vdc.pk1_data import import_archive,go_definitions,prepare_killifish
from vdc.admission import approve_roles, role_manifest


def review(path):
    p=Path(path);policy=read_json(p)
    content={k:v for k,v in policy.items() if k not in {'approval','status'}}
    digest=object_hash(content)
    counts=collections.Counter((r['source'],r['role'],r['split']) for r in policy['samples'])
    rows=''.join('<tr>'+''.join('<td>'+html.escape(str(r.get(k,'')))+'</td>' for k in
        ('sample_key','condition','role','split','biological_unit','cohort_id','parent_sample_key'))+'</tr>' for r in policy['samples'])
    summary=''.join('<li>'+html.escape(' / '.join(key))+': '+str(n)+'</li>' for key,n in sorted(counts.items()))
    document='''<!doctype html><meta charset="utf-8"><title>PK1 私有样本划分审阅</title>
<style>body{font:16px/1.6 system-ui;margin:36px;color:#192b36}table{border-collapse:collapse;font-size:13px}td,th{padding:7px;border:1px solid #ccd5da;text-align:left}code{overflow-wrap:anywhere}h1{font-size:26px}</style>
<h1>PK1 私有样本划分审阅</h1><p>此文件包含内部样本信息，仅供本地/私有服务器审阅，禁止推送公共仓库。</p>
<p>状态：<b>'''+html.escape(policy['status'])+'''</b>。本操作没有进行训练或最终留出评价。</p>
<p>逐行名单内容指纹：<code>'''+digest+'''</code></p>
<ul>'''+summary+'''</ul><p>同一原始 pool 的细胞及派生观测继承同一划分。G1/G2 跨天来自同一培养批次，后续时间点仅作为同批未来检验，不能宣称独立批次泛化。未明确的跨条件培养关系不视为已证实独立。</p>
<p>Single Exit 按预先固定的样本 ID 哈希排序分配，不读取历史 biotime。样本编号不作为时间标签。Late 不作为 Early→Exit 曲线中的中间时间点。</p>
<p>reserved_evaluation 和 temporal_query 仅在模型冻结后使用；prediction_only 无合法 gold 时不计算 accuracy；excluded 不进入拟合。</p>
<table><tr>'''+''.join('<th>'+t+'</th>' for t in ['样本','条件','角色','划分','材料单位','来源队列','时间留出父样本'])+'''</tr>'''+rows+'</table>'
    out=p.with_name('SAMPLE_ROLES_REVIEW.html');out.write_text(document,encoding='utf-8')
    print(json.dumps({'status':policy['status'],'review':str(out),'content_hash':digest,
                      'training_executed':False},indent=2))
    return digest


def main():
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='command',required=True)
    a=sub.add_parser('import');a.add_argument('--archive',type=Path,required=True)
    a.add_argument('--exit-archive',type=Path,required=True);a.add_argument('--private-root',type=Path,required=True)
    a.add_argument('--excluded-exit',nargs='*',default=[])
    a=sub.add_parser('review');a.add_argument('--roles',type=Path,required=True)
    a=sub.add_parser('approve');a.add_argument('--roles',type=Path,required=True)
    a.add_argument('--reviewer',required=True);a.add_argument('--review-hash',required=True)
    a=sub.add_parser('verify');a.add_argument('--roles',type=Path,required=True)
    a=sub.add_parser('restore-approval');a.add_argument('--roles',type=Path,required=True)
    a.add_argument('--approved-copy',type=Path,required=True)
    a=sub.add_parser('prepare');a.add_argument('--private-root',type=Path,required=True)
    a.add_argument('--annotation',type=Path,required=True);a.add_argument('--public-bundle',type=Path,required=True)
    a.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if args.command=='import':
        path=import_archive(args.archive,args.private_root,args.exit_archive,args.excluded_exit);review(path)
    elif args.command=='review':review(args.roles)
    elif args.command=='approve':
        p=read_json(args.roles);digest=object_hash({k:v for k,v in p.items() if k not in {'approval','status'}})
        if digest!=args.review_hash:raise ValueError('Role content differs from the reviewed fingerprint')
        approve_roles(args.roles,args.reviewer);review(args.roles)
    elif args.command=='restore-approval':
        approved,_=role_manifest(args.approved_copy)
        draft=read_json(args.roles)
        content={k:v for k,v in draft.items() if k not in {'approval','status'}}
        if object_hash(content)!=approved['approval']['content_hash']:
            raise ValueError('Imported sources/roles differ from approved local copy; do not replace approval')
        from vdc.io import write_json
        write_json(args.roles,approved);review(args.roles)
    elif args.command=='verify':
        p,rows=role_manifest(args.roles)
        print(json.dumps({'status':'verified','samples':len(rows),'content_hash':p['approval']['content_hash']}))
    else:
        os.environ['VDC_ROLE_MANIFEST']=str((args.private_root/'sample_roles.json').resolve())
        role_manifest()
        if args.out.exists():raise FileExistsError('Prepared outputs are immutable; choose a new output directory')
        original=read_json(args.public_bundle/'manifest.json')['feature_ids']
        shared=[p['id'] for p in go_definitions(args.annotation,original)]
        if not shared:raise ValueError('No shared programme panel')
        from vdc.io import write_json
        write_json(args.out/'panel.json',{'feature_ids':shared,'excluded_from_original':sorted(set(original)-set(shared)),
            'policy':'fixed external annotation minimum three genes; no expression-based selection',
            'original_panel_unchanged':True})
        for modality in ('bulk','core','core_celltypes'):
            print('PREPARE '+modality,flush=True)
            b=prepare_killifish(args.private_root,args.out/modality,args.annotation,shared,modality)
            print(json.dumps({'modality':modality,'observations':len(b.rows),'programmes':len(b.feature_ids),
                              'training_executed':False}),flush=True)


if __name__=='__main__':main()
