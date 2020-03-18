WITH tmptable AS (
	SELECT DISTINCT(reviewerID)
	FROM clothing_interaction15
	WHERE reviewerID NOT IN (SELECT * FROM clothing_interaction15_usertrain) 
	-- LIMIT 2500
    LIMIT 200
	) 
SELECT *
FROM clothing_interaction6 
WHERE reviewerID IN (SELECT * FROM clothing_interaction6_usertrain) 
AND reviewerID NOT IN (SELECT * FROM tmptable) 
ORDER BY reviewerID,rank ASC 
-- LIMIT 10000
;
WITH tmptable AS (
	SELECT DISTINCT(reviewerID)
	FROM clothing_interaction6
	WHERE reviewerID NOT IN (SELECT * FROM clothing_interaction6_usertrain) 
	LIMIT 2500
	) 
SELECT *
FROM clothing_interaction6 
WHERE reviewerID IN (SELECT * FROM tmptable) 
ORDER BY reviewerID,rank ASC ;
;