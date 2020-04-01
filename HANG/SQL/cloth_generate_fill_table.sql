            -- 1st sql is used to COPY `cloth_candidate_asin.sql`
            INSERT INTO clothing_userbase_42_fill_1
            WITH rand_train_set AS ( 
                SELECT DISTINCT(`asin`) FROM clothing_interaction6_itembase 
                ORDER BY RAND(42) 
                LIMIT 15040 	
                ) 
            , tmptable AS ( 
                SELECT DISTINCT(`asin`) 
                FROM clothing_interaction6_itembase 
                WHERE `asin` NOT IN ( 
                    SELECT * FROM rand_train_set 
                    ) 
                LIMIT 2000 
                ) 
            , candidate_set AS ( 
                SELECT clothing_interaction6_itembase.ID, clothing_interaction6_itembase.reviewerID, clothing_interaction6_itembase.`asin` 
                FROM clothing_interaction6_itembase 
                WHERE clothing_interaction6_itembase.`asin` IN ( 
                    SELECT * FROM tmptable 
                ) 
                AND rank = 6 
                ORDER BY `asin`,rank ASC 
            )
            SELECT RANK() OVER (PARTITION BY reviewerID ORDER BY unixReviewTime,ID ASC) AS rank, 
            clothing_review.`ID`, clothing_review.reviewerID , clothing_review.`asin`, 
            clothing_review.overall, clothing_interaction6_rm_sw.reviewText, clothing_review.unixReviewTime 
            FROM  clothing_review , clothing_interaction6_rm_sw 
            WHERE reviewerID IN (SELECT reviewerID FROM candidate_set) 
            AND clothing_review.ID = clothing_interaction6_rm_sw.ID 
            ORDER BY reviewerID,unixReviewTime ASC 
            ;

            -- 2nd sql is used to drop rank higher than k
            DELETE FROM clothing_userbase_42_fill_1
            WHERE rank>6;

            -- 3rd sql is used to set NULL value to frame that want to generate.
            UPDATE clothing_userbase_42_fill_1 
            SET reviewText=NULL, unixReviewTime=NULL
            WHERE rank = 1
            ;

            -- 4th sql is used to set generate reviews.
            UPDATE clothing_userbase_42_fill_1 , clothing_sparsity_generation_res_42
            SET clothing_userbase_42_fill_1.reviewText = clothing_sparsity_generation_res_42.generative_review 
            WHERE clothing_userbase_42_fill_1.reviewerID = clothing_sparsity_generation_res_42.reviewerID 
            AND clothing_userbase_42_fill_1.`asin` = clothing_sparsity_generation_res_42.`asin` 
            ;

            SELECT * FROM clothing_userbase_42_fill_1
            WHERE rank=1
            ;