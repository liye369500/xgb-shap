import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, plot_tree
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import shap


# 读取数据并准备自变量和因变量
def get_data(file_path, feature_columns):
    data = pd.read_excel(file_path)
    # 确保 feature_columns 中的列名存在于 data 中
    feature_columns = [col for col in feature_columns if col in data.columns]
    x = data[feature_columns]  # 使用提供的列名列表作为自变量
    y = data[['HVI']]  # 因变量只包括 'LST'
    return x, y


# 检查并清理数据
def clean_data(x, y):
    # 填充NaN值
    x.fillna(x.mean(), inplace=True)
    y.fillna(y.mean(), inplace=True)

    # 移除无穷值
    x = x[~np.isinf(x).all(axis=1)]
    y = y[~np.isinf(y).any(axis=1)]

    return x, y


# 定义XGBoost回归模型训练和评估的函数
def train_evaluate_xgb(x, y, best_param, y_name):
    x, y = clean_data(x, y)  # 确保数据是干净的
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
    xgb = XGBRegressor(**best_param, use_label_encoder=False, eval_metric='logloss')
    xgb.fit(x_train, y_train)
    y_predict = xgb.predict(x_test)
    # 计算R2分数和MSE
    r2_score_value = r2_score(y_test, y_predict)  # 使用r2_score函数
    mse = mean_squared_error(y_test, y_predict)

    print(f"XGBoost的R2值为（对于 {y_name}）: {r2_score_value}")
    print(f"XGBoost的MSE的值为（对于 {y_name}）: {mse}")

    # 使用 SHAP 库解释模型
    explainer = shap.Explainer(xgb)
    # 设置全局的图表尺度和字体大小

    # 计算 SHAP 值，这里需要传递训练数据 x_train 和标签 y_train
    shap_values = explainer(x_train, y_train)
    # 绘制全局条形图
    shap.plots.bar(shap_values, max_display=15, show=False)
    plt.savefig("shap_summary.png", dpi=300, bbox_inches='tight')
    # 显示图表

    plt.show()

    # 使用 SHAP 库解释模型
    explainer = shap.Explainer(xgb)
    # 计算 SHAP 值，这里需要传递训练数据 x_train 和标签 y_train
    shap_values = explainer(x_train, y_train)

    shap.plots.violin(shap_values, plot_size=0.45, max_display=14, show=False)
    plt.savefig("shap_violin.png", dpi=300, bbox_inches='tight')

    # 显示图表
    plt.show()

    # 假设 y_test 和 y_predict 是 NumPy 数组

    plt.figure(figsize=(12, 12))

    # 绘制实际值和预测值的散点图
    plt.scatter(y_test, y_predict, color='red', s=180, label='Actual Values', alpha=0.8)  # 实际值点
    plt.scatter(y_predict, y_test, color='blue', s=180, label='Prediction', alpha=0.8)  # 预测值点

    # 添加图例，并将其放在图表下方
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2)

    # 设置图表标题和轴标签
    plt.title('Prediction vs Actual Values')
    plt.xlabel('Predicted Values')
    plt.ylabel('Actual Values')

    # 绘制红色实线对角线，表示完美预测
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')

    # 添加网格线
    plt.grid(True)

    # 保存图表
    plt.savefig("Prediction.png", dpi=300, bbox_inches='tight')  # 修正了这里的语法错误

    # 显示图表
    plt.show()


    return xgb



# 使用GridSearchCV进行参数调优
def grid_search_optimize(x, y, y_name):
    # 定义参数网格
    param_grid = {
        'n_estimators': [70],
        'learning_rate': [0.1],
        'max_depth': [10],
        'subsample': [1],
        'colsample_bytree': [0.9],
        'reg_lambda': [0.4],
        'reg_alpha': [0.2],

    }

    # 使用 GridSearchCV 进行参数调优
    xgb = XGBRegressor(use_label_encoder=False, eval_metric='logloss')
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1)
    grid_search.fit(x, y[y_name])

    # 输出结果
    print("参数的最佳取值：", grid_search.best_params_)
    print("最佳模型得分:", -grid_search.best_score_)
    return grid_search.best_params_

def print_feature_importances(xgb_models, feature_names):
    for i, model in enumerate(xgb_models):
        importances = model.feature_importances_
        sorted_idx = importances.argsort()[::-1]
        print(f"\n特征重要性排名（模型 {i + 1}）:")
        for fi in sorted_idx:
            print(f"{feature_names[fi]:<25} : {importances[fi]:.8f}")


# 打印特征贡献度表格
def task_three(file_path, feature_columns):
    x, y = get_data(file_path, feature_columns)
    xgb_models = []  # 初始化模型列表

    # 确保 'LST' 列存在于 y 中
    if 'HVI' in y.columns:
        y_name = 'HVI'
        best_params = grid_search_optimize(x, y, y_name)  # 假设这个函数已正确定义
        xgb_model = train_evaluate_xgb(x, y, best_params, y_name)
        xgb_models.append(xgb_model)  # 将模型添加到列表中

        # 打印特征重要性
        print_feature_importances(xgb_models, x.columns)
    else:
        print("'HVI' not found in y columns. Please check the data and column names.")


# 设置 输入文件，输入特征x

# 确保 feature_columns 列表中的列名没有多余空格，并且与 Excel 文件中的列名完全一致

feature_columns = ['NDVI', 'MNDWI', 'NDBI', 'Elevation', 'H', 'AR', ]#输入自定义的指标
task_three(file_path=r'C:/Users/Desktop/no.xlsx', feature_columns=feature_columns)