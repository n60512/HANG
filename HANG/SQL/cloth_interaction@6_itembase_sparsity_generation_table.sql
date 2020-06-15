WITH rand_train_set AS (
	SELECT DISTINCT(`asin`) FROM clothing_interaction6_itembase
	ORDER BY RAND() 
	LIMIT 12800 
	) 
SELECT clothing_interaction6_itembase.rank, clothing_interaction6_itembase.ID, clothing_interaction6_itembase.reviewerID, clothing_interaction6_itembase.`asin`, 
clothing_interaction6_itembase.overall, clothing_interaction6_itembase.reviewText, clothing_interaction6_itembase.unixReviewTime
FROM clothing_interaction6_itembase, clothing_interaction6_rm_sw 
WHERE clothing_interaction6_itembase.`asin` IN ( 
	SELECT * FROM rand_train_set 
) 
AND clothing_interaction6_itembase.ID = clothing_interaction6_rm_sw.ID 
ORDER BY `asin`,rank ASC 
;
SELECT * FROM clothing_sparsity_generation_oringial_0527;