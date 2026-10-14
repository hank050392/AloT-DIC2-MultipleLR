# Regression Analysis Explorer

GitHub Pages 版本的迴歸分析網站，支援：

- 50 Startups
- Boston Housing
- Custom CSV URL
- Multiple Linear Regression (OLS)
- Feature Progression / Exhaustive Search
- k = 1 ~ 5 Feature 比較
- RMSE
- R²
- Adjusted R²
- MAE
- Regression Equation
- Live Prediction
- Residual Table

## Run locally

直接用瀏覽器開啟 `index.html` 即可；若瀏覽器限制跨網域 CSV 載入，建議使用 VS Code Live Server。

## GitHub Pages

1. Push 此專案到 GitHub repository。
2. GitHub repository → **Settings** → **Pages**。
3. Source 選 **Deploy from a branch**。
4. Branch 選 `main`，Folder 選 `/ (root)`。
5. 儲存後等待 GitHub Pages 完成部署。

網址通常會是：

`https://<username>.github.io/<repository-name>/`

## Notes

Custom Dataset URL 必須：
- 是可直接讀取的 CSV 網址
- 允許瀏覽器跨網域讀取（CORS）
- Target 必須是數值欄位
- Feature Progression 目前只分析數值型 Feature


## v2
- Added automatic one-hot encoding for categorical features.
- 50 Startups now produces 5 model features: 3 numeric + 2 State dummy variables.
- Feature Progression can therefore run k=1..5 like the class example.

## v3
Added real browser-side feature-selection methods:
- Lasso ranking (coordinate descent)
- Forward Sequential Feature Selection
- Recursive Feature Elimination
- ElasticNet ranking
- Exhaustive Search
- Compare All 5 Methods table

## v4
Added statistical inference:
- F-statistic and model p-value
- Standard error
- t-statistic
- coefficient p-value
- 95% confidence interval
- VIF

Also added `.github/workflows/pages.yml` for GitHub Actions deployment to GitHub Pages.
