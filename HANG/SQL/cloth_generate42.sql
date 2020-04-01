WITH rand_train_set AS ( 
	SELECT DISTINCT(`asin`) FROM clothing_interaction6_itembase 
	ORDER BY RAND() 
	LIMIT 15040
	-- LIMIT 14000 
	) 
SELECT clothing_interaction6_itembase.rank, clothing_interaction6_itembase.ID, clothing_interaction6_itembase.reviewerID, clothing_interaction6_itembase.`asin`, 
clothing_interaction6_itembase.overall, clothing_interaction6_rm_sw.reviewText, clothing_interaction6_itembase.unixReviewTime 
FROM clothing_interaction6_itembase, clothing_interaction6_rm_sw 
WHERE clothing_interaction6_itembase.`asin` IN ( 
	SELECT * FROM rand_train_set 
) 
AND clothing_interaction6_itembase.ID = clothing_interaction6_rm_sw.ID 
ORDER BY `asin`,rank ASC 
-- LIMIT 10000
;
SELECT * FROM clothing_sparsity_trainset_42 
LIMIT 61440	-- 10240 * 6
;
;
-- 15000:2000 = 7.5:1