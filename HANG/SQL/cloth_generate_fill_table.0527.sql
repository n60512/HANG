INSERT INTO 
clothing_userbase0527_42_fill_4
SELECT * 
FROM 
clothing_userbase_42_oringinal
;	-- 自原始 userbase set 提取資料集

UPDATE clothing_userbase0527_42_fill_4 
SET reviewText = NULL, unixReviewTime=NULL
WHERE rank = 4
;	-- 刪除欲生成評論位置

UPDATE clothing_userbase0527_42_fill_4, clothing_sparsity_generation_res_grm_0527 SET
clothing_userbase0527_42_fill_4.reviewText = clothing_sparsity_generation_res_grm_0527.generative_review
WHERE clothing_userbase0527_42_fill_4.`asin` = clothing_sparsity_generation_res_grm_0527.`asin`
AND clothing_userbase0527_42_fill_4.reviewerID = clothing_sparsity_generation_res_grm_0527.reviewerID
AND rank = 4
;	-- 自生成資料表取評論填補

SELECT * FROM 
clothing_userbase0527_42_fill_4
WHERE rank = 4;