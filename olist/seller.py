
import pandas as pd
import numpy as np
from olist.data import Olist
from olist.order import Order


class Seller:
    def __init__(self):
        # Import data only once
        olist = Olist()
        self.data = olist.get_data()
        self.order = Order()

    def get_seller_features(self):
        """
        Returns a DataFrame with:
        'seller_id', 'seller_city', 'seller_state'
        """
        sellers = self.data['sellers'].copy(
        )  # Make a copy before using inplace=True so as to avoid modifying self.data
        sellers.drop('seller_zip_code_prefix', axis=1, inplace=True)
        sellers.drop_duplicates(
            inplace=True)  # There can be multiple rows per seller
        return sellers

    def get_seller_delay_wait_time(self):
        """
        Returns a DataFrame with:
        'seller_id', 'delay_to_carrier', 'wait_time'
        """
        # Get data
        order_items = self.data['order_items'].copy()
        orders = self.data['orders'].query("order_status=='delivered'").copy()

        ship = order_items.merge(orders, on='order_id')

        # Handle datetime
        ship.loc[:, 'shipping_limit_date'] = pd.to_datetime(
            ship['shipping_limit_date'])
        ship.loc[:, 'order_delivered_carrier_date'] = pd.to_datetime(
            ship['order_delivered_carrier_date'])
        ship.loc[:, 'order_delivered_customer_date'] = pd.to_datetime(
            ship['order_delivered_customer_date'])
        ship.loc[:, 'order_purchase_timestamp'] = pd.to_datetime(
            ship['order_purchase_timestamp'])

        # Compute delay and wait_time
        def delay_to_logistic_partner(d):
            days = np.mean(
                (d.order_delivered_carrier_date - d.shipping_limit_date) /
                np.timedelta64(24, 'h'))
            if days > 0:
                return days
            else:
                return 0

        def order_wait_time(d):
            days = np.mean(
                (d.order_delivered_customer_date - d.order_purchase_timestamp)
                / np.timedelta64(24, 'h'))
            return days

        delay = ship.groupby('seller_id')\
                    .apply(delay_to_logistic_partner)\
                    .reset_index()

        delay.columns = ['seller_id', 'delay_to_carrier']

        wait = ship.groupby('seller_id')\
                   .apply(order_wait_time)\
                   .reset_index()
        wait.columns = ['seller_id', 'wait_time']

        df = delay.merge(wait, on='seller_id')

        return df

    def get_active_dates(self):
        """
        Returns a DataFrame with:
        'seller_id', 'date_first_sale', 'date_last_sale', 'months_on_olist'
        """
        # First, get only orders that are approved
        orders_approved = self.data['orders'][[
            'order_id', 'order_approved_at'
        ]].dropna()

        # Then, create a (orders <> sellers) join table because a seller can appear multiple times in the same order
        orders_sellers = orders_approved.merge(self.data['order_items'],
                                               on='order_id')[[
                                                   'order_id', 'seller_id',
                                                   'order_approved_at'
                                               ]].drop_duplicates()
        orders_sellers["order_approved_at"] = pd.to_datetime(
            orders_sellers["order_approved_at"])

        # Compute dates
        orders_sellers["date_first_sale"] = orders_sellers["order_approved_at"]
        orders_sellers["date_last_sale"] = orders_sellers["order_approved_at"]
        df = orders_sellers.groupby('seller_id').agg({
            "date_first_sale": min,
            "date_last_sale": max
        })
        df['months_on_olist'] = round(
            (df['date_last_sale'] - df['date_first_sale']) /
            np.timedelta64(1, 'M'))
        return df

    def get_quantity(self):
        """
        Returns a DataFrame with:
        'seller_id', 'n_orders', 'quantity', 'quantity_per_order'
        """
        order_items = self.data['order_items']

        n_orders = order_items.groupby('seller_id')['order_id']\
            .nunique()\
            .reset_index()
        n_orders.columns = ['seller_id', 'n_orders']

        quantity = order_items.groupby('seller_id', as_index=False).agg(
            {'order_id': 'count'})
        quantity.columns = ['seller_id', 'quantity']

        result = n_orders.merge(quantity, on='seller_id')
        result['quantity_per_order'] = result['quantity'] / result['n_orders']
        return result

    def get_sales(self):
        """
        Returns a DataFrame with:
        'seller_id', 'sales'
        """
        return self.data['order_items'][['seller_id', 'price']]\
            .groupby('seller_id')\
            .sum()\
            .rename(columns={'price': 'sales'})

    def get_review_score(self):
        """
        Returns a DataFrame with:
        'seller_id', 'share_of_five_stars', 'share_of_one_stars', 'review_score'
        """

        orders=self.data['orders']
        order_items=self.data['order_items']
        order_reviews=self.data['order_reviews']

        order_and_seller = pd.merge(left =order_items, right=orders, on='order_id', how='inner')
        order_and_seller = order_and_seller[['order_id', 'seller_id']]

        order_and_seller_and_review = pd.merge(left=order_and_seller, right=order_reviews, on='order_id', how='inner')
        order_and_seller_and_review = order_and_seller_and_review[['order_id', 'seller_id', 'review_score']]

        average_review_score = order_and_seller_and_review.groupby('seller_id').agg({'review_score':'mean'})
        average_review_score = average_review_score.reset_index()

        total_review_score = order_and_seller_and_review.groupby('seller_id').agg({'review_score':'count'})
        total_review_score = total_review_score.rename(columns={'review_score':'total_reviews'})
        total_review_score = total_review_score.reset_index()

        one_star_review_score = order_and_seller_and_review[order_and_seller_and_review['review_score'] ==1].groupby('seller_id').agg({'review_score':'count'})
        one_star_review_score = one_star_review_score.rename(columns={'review_score':'one_star_review_score'})
        one_star_review_score = one_star_review_score.reset_index()

        five_star_review_score = order_and_seller_and_review[order_and_seller_and_review['review_score'] ==5].groupby('seller_id').agg({'review_score':'count'})
        five_star_review_score = five_star_review_score.rename(columns={'review_score':'five_star_review_score'})
        five_star_review_score = five_star_review_score.reset_index()

        all_df_review_score = pd.merge(left=average_review_score, right=one_star_review_score, on='seller_id', how='left')
        all_df_review_score = pd.merge(left=all_df_review_score, right=five_star_review_score, on='seller_id', how='left')
        all_df_review_score = pd.merge(left=all_df_review_score, right=total_review_score, on='seller_id', how='left')
        all_df_review_score

        all_df_review_score['share_of_one_stars'] = all_df_review_score['one_star_review_score'] / all_df_review_score['total_reviews']
        all_df_review_score['share_of_five_stars'] = all_df_review_score['five_star_review_score'] / all_df_review_score['total_reviews']

        get_review_score = all_df_review_score[['seller_id', 'share_of_one_stars', 'share_of_five_stars', 'review_score']]
        get_review_score = get_review_score.drop_duplicates()
        return get_review_score

    def get_training_data(self):
        """
        Returns a DataFrame with:
        ['seller_id', 'seller_city', 'seller_state', 'delay_to_carrier',
        'wait_time', 'date_first_sale', 'date_last_sale', 'months_on_olist', 'share_of_one_stars',
        'share_of_five_stars', 'review_score', 'n_orders', 'quantity',
        'quantity_per_order', 'sales']
        """

        training_set =\
            self.get_seller_features()\
                .merge(
                self.get_seller_delay_wait_time(), on='seller_id'
               ).merge(
                self.get_active_dates(), on='seller_id'
               ).merge(
                self.get_quantity(), on='seller_id'
               ).merge(
                self.get_sales(), on='seller_id'
               )

        if self.get_review_score() is not None:
            training_set = training_set.merge(self.get_review_score(),
                                              on='seller_id')

        return training_set

    def get_training_profit_data(self):
        """
        Returns a DataFrame with:
        ['seller_id', 'revenues', 'cost_of_reviews', 'profits']
        """
        sellers_df = self.get_training_data().copy()

        order_items = self.data['order_items'].copy()
        order_reviews = self.data['order_reviews'].copy()

        # Sales fees
        sellers_df['sales_fees'] = 0.10*(sellers_df['sales'])

        # Subscription fees
        sellers_df['subscription_fees'] = sellers_df['months_on_olist'] * 80

        # Total revenues
        sellers_df['revenues'] = sellers_df['sales_fees'] + sellers_df['subscription_fees']
        sellers_df_revenues = sellers_df[['seller_id', 'revenues']].copy()

        # Get review score for each seller id for each order id
        order_items_with_review = pd.merge(left=order_items, right=order_reviews, on='order_id', how='left')
        order_items_with_review = order_items_with_review[['order_id', 'seller_id', 'review_score']]
        order_items_with_review = order_items_with_review.dropna()

        # Map the review score with the cost associated
        reputation_costs = {1: 100, 2: 50, 3: 40, 4: 0, 5: 0}

        def review_score_mapping(int):
            return reputation_costs[int]

        order_items_with_review['review_score'] = order_items_with_review['review_score'].astype(np.int64)
        order_items_with_review['reputation_cost'] = order_items_with_review['review_score'].apply(lambda x: review_score_mapping(x))

        seller_id_reputation_costs = order_items_with_review.groupby('seller_id').agg({'reputation_cost':'sum'})
        seller_id_reputation_costs = seller_id_reputation_costs.reset_index()
        seller_id_reputation_costs = seller_id_reputation_costs.rename(columns={'reputation_cost':'cost_of_reviews'})

        # Profit
        sellers_profit = pd.merge(left=sellers_df_revenues, right=seller_id_reputation_costs, on='seller_id', how='left')
        sellers_profit['profits'] = sellers_profit['revenues'] - sellers_profit['cost_of_reviews']

        return sellers_profit
