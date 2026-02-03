
'''
load args from args.py and model.yaml-->return args (SimpleNamespace)
'''


from tabulate import tabulate

from utils import read_yaml_to_dict, setup_logger
from .args import Prune_Args


def get_args(cfg, table=True):
    '''
    cfg (str | Path | SimpleNamespace): Configuration data source-->model.yaml or SimpleNamespace object
    '''
    cfg = read_yaml_to_dict(cfg)
    args = Prune_Args(cfg).args
    if not args.exp_name: # llama-2-13b_wanda_unstructured_sp0.6
        if args.prune_m == 0 or args.prune_n == 0:
            args.exp_name = args.model.split('/')[-1]+'_'+args.prune_method+'_'+ args.sparsity_type + '_'+ 'sp'+str(args.sparsity_ratio)
        else:
            args.exp_name = args.model.split('/')[-1]+'_'+args.prune_method+'_'+'n'+str(args.prune_n) +'m'+str(args.prune_m)
    args.logger= setup_logger(args.exp_name, args.output_dir)
    if table:
        table_data = [["param", "value", "description"]]
        args.logger.info("|      Parameters      |      Value     |     Description     |")
        for key, value in vars(args).items():
            # 查找对应的描述
            description = "None"
            for action in Prune_Args(cfg).parser._actions:
                if key == action.dest:
                    description = action.help
                    break
            args.logger.info(f"|{key}|{value}|{description}|")
            table_data.append([f"--{key}", value, description])
        table_str = tabulate(table_data, headers="firstrow", tablefmt="grid")
        # args.logger.info("\n" +table_str)
    else:
        args.logger.info(args)
    return args


