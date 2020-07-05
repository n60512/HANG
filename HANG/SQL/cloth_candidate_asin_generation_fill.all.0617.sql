WITH rand_train_set AS ( 
	SELECT DISTINCT(`asin`) FROM clothing_interaction6_itembase 
	ORDER BY RAND() 
	LIMIT 12800 
	) 
, candidate_set AS ( 
	SELECT clothing_interaction6_itembase.ID, clothing_interaction6_itembase.reviewerID, clothing_interaction6_itembase.`asin` 
	FROM clothing_interaction6_itembase 
	WHERE clothing_interaction6_itembase.`asin` IN ( 
		SELECT * FROM rand_train_set 
	) 
	AND rank = 6 
	ORDER BY `asin`,rank ASC 
)
SELECT RANK() OVER (PARTITION BY reviewerID ORDER BY unixReviewTime,ID ASC) AS rank, 
clothing_review.`ID`, clothing_review.reviewerID , clothing_review.`asin`, 
clothing_review.overall, clothing_interaction6_without_rm_sw.reviewText, clothing_review.unixReviewTime 
FROM  clothing_review , clothing_interaction6_without_rm_sw 
WHERE reviewerID IN (SELECT reviewerID FROM candidate_set) 
AND clothing_review.ID = clothing_interaction6_without_rm_sw.ID 
ORDER BY reviewerID,unixReviewTime ASC 
;
SELECT * FROM clothing_userbase0617_42_fill_all
ORDER BY reviewerID, rank;