"""C4 归因：签名残差（declared + unknown）逐条归到有名边界。非判据，产出 logs/attr.json 与汇总。"""
import sys, json, collections, re
sys.path.insert(0, 'model')
import odsjson as O, odssig as G, cppverify as CV
BUCKETS = [   # (边界名, 不支持名字的正则)；按顺序取第一个命中
 ('父 op / 区域 / 块结构', r'getParentOp|getParentOfType|getParentWithTrait|getRegion|getBody|getBlock|Block\b|front\(\)|back\(\)|getTerminator|getRegions|walk|getOps|getNumRegions|getSuccessor|getArgument'),
 ('符号表 / 符号引用', r'Symbol|lookup|getCallableRegion|FunctionOpInterface|getFunction'),
 ('操作数的值 / 常量 / 定义 op（值层，已冻结）', r'matchPattern|m_Constant|getDefiningOp|getConstant|ConstantIntValue|getConstantInt|isConstant|getUses|getUsers|use_empty|hasOneUse|getValue\b'),
 ('DataLayout', r'DataLayout|getTypeSize|getAlignment'),
 ('方言外部类型 / 接口实现（开放世界）', r'Interface|isa<.*Interface|TypeInterface|dyn_cast<.*Interface'),
]
out = {}; tot = collections.Counter(); byb = collections.Counter(); per = collections.defaultdict(list)
for dia in O.DIALECTS:
    try: names = list(O.ops(dia))
    except Exception as e: tot['方言加载失败'] += 1; continue
    for op in names:
        try: S = G.lookup(op); S = S[1] if S else None
        except Exception as e: S = None
        if S is None: tot['op 不可查'] += 1; continue
        tot['op'] += 1
        res = list(S['declared']) + [u if isinstance(u, str) else str(u) for u in S['unknown']]
        if not res: tot['残差为空'] += 1; continue
        for d in res:
            tot['条目'] += 1
            which = 'verify' if d == 'C++ verify()' else 'infer' if d in ('C++ inferReturnTypes', '接口 verify: InferTypeOpInterface::Trait') else ('iface:' + d.split(': ', 1)[1]) if d.startswith('接口 verify: ') else None
            if d == 'C++ verifyRegions()': b = '区域 verify（verifyRegions，区域结构）'
            elif d.startswith('开放接口 isa'): b = '方言外部类型 / 接口实现（开放世界）'
            elif d.startswith('端口类型原子'): b = 'ODS 类型 CPred 未编译'
            elif which:
                try: ok, bad = CV.completeness(dia, op, which)
                except Exception as e: ok, bad = False, {f'完备性检查异常: {type(e).__name__}'}
                if ok: b = 'C++ 完备但仍在残差（需查）'
                else:
                    txt = ' '.join(sorted(bad)); b = None
                    if '<无源码>' in txt: b = '无 C++ 源码（未下载 / 生成代码）'
                    for nm, rx in BUCKETS:
                        if b is None and re.search(rx, txt): b = nm
                    if b is None: b = '解释器词汇未覆盖（可扩）'
                    per[b].append((op, d, txt[:160]))
            elif d.startswith('native trait '): b = 'native trait 未建模'
            elif d.startswith('符号使用义务'): b = '符号使用义务（verifySymbolUses，symtab.USES 未登记）'
            elif d.startswith('PredOpTrait'): b = 'PredOpTrait 的 $_op CPred 未编译（跨端口谓词）'
            elif d.startswith('属性约束'): b = '属性 CPred 未编译'
            elif d.startswith('TypesMatchWith'): b = 'TypesMatchWith 变换未解释'
            else: b = '未分类：' + d[:40]
            if op.startswith('transform.'): b = 'Transform 方言（元编程 op：' + b + '）'
            byb[b] += 1
            out.setdefault(op, []).append((d, b))
json.dump({'tot': tot, 'byb': byb, 'ops': out, 'ex': {k: v[:30] for k, v in per.items()}}, open('logs/attr.json', 'w'), ensure_ascii=False, indent=1)
print(dict(tot)); [print(f'{v:5d}  {k}') for k, v in byb.most_common()]
for k, v in per.items():
    print('──', k); [print('   ', x) for x in v[:6]]
