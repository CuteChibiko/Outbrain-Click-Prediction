#ifndef _LIBFFM_H
#define _LIBFFM_H

#ifdef __cplusplus
extern "C" 
{

namespace ffm
{
#endif

typedef float ffm_float;
typedef double ffm_double;
typedef int ffm_int;
typedef long long ffm_long;

struct ffm_node
{
    ffm_int f; //フィールド番号
    ffm_int j; //素性番号
    ffm_float v; //値
};

struct ffm_problem
{
    ffm_int n; //素性の数
    ffm_int l; //train,valid,testの件数
    ffm_int m; //フィールドの数
    ffm_node *X; //ffm_nodeのリスト
    ffm_long *P; //Xのインデックス
    ffm_float *Y;//ラベル
    ffm_int *map_j2f; //素性からフィールド番号へのマップ
    ffm_int group_line; //グループ数
    ffm_int *group_counts; //グループ内のレコード数
    ffm_int *group_cumsum; //グループの累積レコード数(そのグループのレコード数も含む
    ffm_float *data_weight; //各サンプル点のウェイト（トレーニングデータでのみ利用）
};

struct ffm_model
{
    ffm_int n; //素性の数?
    ffm_int m; //フィールドの数?
    ffm_int k;
    ffm_float *W;
    bool normalization;
};

struct ffm_parameter
{
    ffm_float eta;
    ffm_float lambda;
    ffm_int nr_iters;
    ffm_int k;
    ffm_int nr_threads;
    bool quiet;
    bool normalization;
    bool random;
    bool auto_stop;
    bool use_default_train;
};

ffm_problem* ffm_read_problem(char const *path, char const *group_path, char const *weight_path);

int ffm_read_problem_to_disk(char const *txt_path, char const *bin_path);

void ffm_destroy_problem(struct ffm_problem **prob);

ffm_int ffm_save_model(ffm_model *model, char const *path);

ffm_model* ffm_load_model(char const *path);

void ffm_destroy_model(struct ffm_model **model);

ffm_parameter ffm_get_default_param();

ffm_model* ffm_train_with_validation(struct ffm_problem *Tr, struct ffm_problem *Va, struct ffm_parameter param);

ffm_model* ffm_train(struct ffm_problem *prob, struct ffm_parameter param);

ffm_model* ffm_train_with_validation_on_disk(char const *Tr_path, char const *Va_path, struct ffm_parameter param);

ffm_model* ffm_train_on_disk(char const *path, struct ffm_parameter param);

ffm_float ffm_predict(ffm_node *begin, ffm_node *end, ffm_model *model);

ffm_float ffm_cross_validation(struct ffm_problem *prob, ffm_int nr_folds, struct ffm_parameter param);

#ifdef __cplusplus
} // namespace ffm

} // extern "C"
#endif

#endif // _LIBFFM_H
