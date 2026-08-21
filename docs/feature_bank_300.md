# Feature Bank 300

Цель: собрать 300 осмысленных табличных фичей, которые описывают не просто объем активности, а пользовательские паттерны: ритм, намерение, трение в воронке, уход после покупки, сезонность внутри истории, стабильность чека и жизненный цикл.

Правила для реализации:
- каждая фича считается только по `event_date < cutoff`;
- таргет и данные после cutoff не используются;
- фичи добавляются новыми колонками, старые не переписываются;
- базовый feature set для развития: `long_buy_post_order`;
- окна использовать осмысленно: `14`, `30`, `60`, `90`, `120`, `180`, `365`, без перебора `30/29/28/...`;
- сначала внедрять батчами по 40-60 фичей, затем отбирать по CV, gain, permutation importance и SHAP.

## 1. Purchase Cadence, 15

1. `order_gap_mean_all` - средний интервал между днями с покупкой.
2. `order_gap_median_all` - медианный интервал между днями с покупкой.
3. `order_gap_p90_all` - 90-й перцентиль интервалов между покупками.
4. `order_gap_cv_all` - `std(order_gap) / mean(order_gap)`.
5. `order_gap_cv_recent90` - вариативность интервалов покупок, если обе покупки внутри последних 90 дней.
6. `order_gap_last_over_mean` - последний интервал без покупки / средний интервал покупок.
7. `order_gap_last_over_median` - последний интервал без покупки / медианный интервал покупок.
8. `order_gap_accel_last2_vs_prev2` - отношение среднего из двух последних gap к двум предыдущим.
9. `order_gap_shortest_all` - самый короткий интервал между покупками.
10. `order_gap_longest_all` - самый длинный интервал между покупками.
11. `order_gap_entropy_bucketed` - энтропия gap-бакетов `1-3`, `4-7`, `8-14`, `15-30`, `31+`.
12. `order_regularity_score` - `1 / (1 + order_gap_cv_all)`.
13. `order_cycle_phase` - `recency_to_ord_days / (order_gap_median_all + 1)`.
14. `expected_next_order_overdue_days` - `recency_to_ord_days - order_gap_median_all`.
15. `expected_next_order_overdue_ratio` - `expected_next_order_overdue_days / (order_gap_p90_all + 1)`.

## 2. Active-Day Rhythm, 15

16. `active_gap_mean_all` - средний интервал между любыми активными днями.
17. `active_gap_median_all` - медианный интервал между активными днями.
18. `active_gap_p90_all` - 90-й перцентиль интервалов активности.
19. `active_gap_cv_all` - вариативность интервалов активности.
20. `active_gap_last_over_mean` - текущая пауза активности / средний gap активности.
21. `active_streak_current` - длина текущей серии активных дней прямо перед cutoff.
22. `active_streak_max_120` - максимальная серия активных дней за 120 дней.
23. `active_streak_mean_120` - средняя длина серии активных дней за 120 дней.
24. `inactive_streak_max_120` - максимальная серия неактивных дней между активностями за 120 дней.
25. `active_week_entropy_120` - энтропия распределения активных дней по неделям за 120 дней.
26. `active_burst_count_120` - число бурстов активности, где между активными днями gap не больше 2 дней.
27. `active_burst_mean_len_120` - средняя длина таких бурстов.
28. `active_burst_max_len_120` - максимальная длина бурста активности.
29. `active_burst_gap_mean_120` - средний интервал между бурстами.
30. `active_rhythm_score` - `active_streak_mean_120 / (active_gap_cv_all + 1)`.

## 3. Search Intent Shape, 15

31. `searches_per_search_day_30` - поисков на день с поиском за 30 дней.
32. `searches_per_search_day_90` - поисков на день с поиском за 90 дней.
33. `search_depth_ratio_14_to_90` - интенсивность поиска за 14 дней / 90 дней.
34. `search_day_share_active_30` - доля активных дней, где был поиск, за 30 дней.
35. `search_day_share_active_90` - доля активных дней, где был поиск, за 90 дней.
36. `search_zero_after_active_30` - активные дни без поиска за 30 дней.
37. `search_zero_after_active_share_30` - доля активных дней без поиска за 30 дней.
38. `search_spike_ratio_30` - максимум дневных searches за 30 / среднее за search-day.
39. `search_spike_ratio_90` - максимум дневных searches за 90 / среднее за search-day.
40. `search_last7_share_30` - доля searches последних 7 дней от searches за 30.
41. `search_last14_share_90` - доля searches последних 14 дней от searches за 90.
42. `search_acceleration_14_60` - дневная интенсивность поиска `14d / 60d`.
43. `search_decay_since_peak_90` - дней от последнего локального пика searches за 90.
44. `search_peak_recentness_score_90` - `1 / (1 + search_decay_since_peak_90)`.
45. `search_intent_score` - `log1p(searches_30) * search_day_share_active_30 / (recency_search_days + 1)`.

## 4. Cart Friction, 15

46. `cart_days_without_order_30` - дни с cart, но без order, за 30 дней.
47. `cart_days_without_order_90` - дни с cart, но без order, за 90 дней.
48. `cart_no_order_share_30` - `cart_days_without_order_30 / cart_days_30`.
49. `cart_no_order_share_90` - `cart_days_without_order_90 / cart_days_90`.
50. `cart_to_order_same_day_rate_90` - доля cart-дней, где в тот же день был order.
51. `cart_to_order_same_day_rate_all` - same-day cart to order по всей истории.
52. `last_cart_is_stale` - `recency_to_cart_days < recency_to_ord_days` и cart старше 14 дней.
53. `recent_cart_no_buy_flag_14` - cart был за 14 дней, order не было за 14 дней.
54. `cart_after_last_order_days` - число дней с cart после последней покупки.
55. `cart_after_last_order_share` - `cart_after_last_order_days / recency_to_ord_days`.
56. `cart_pressure_30` - `log1p(to_cart_30) / (recency_to_cart_days + 1)`.
57. `cart_pressure_90` - `log1p(to_cart_90) / (recency_to_cart_days + 1)`.
58. `cart_search_gap_abs` - `abs(recency_to_cart_days - recency_search_days)`.
59. `cart_search_gap_signed` - `recency_to_cart_days - recency_search_days`.
60. `cart_friction_score` - `cart_no_order_share_90 * log1p(to_cart_90)`.

## 5. Search-to-Order Journey, 15

61. `search_to_order_same_day_rate_30` - same-day search to order за 30 дней.
62. `search_to_order_same_day_rate_90` - same-day search to order за 90 дней.
63. `search_to_cart_same_day_rate_30` - same-day search to cart за 30 дней.
64. `search_to_cart_same_day_rate_90` - same-day search to cart за 90 дней.
65. `search_days_before_order_mean_90` - среднее число search-days за 7 дней перед order-day.
66. `searches_before_order_mean_90` - среднее число searches за 7 дней перед order-day.
67. `cart_days_before_order_mean_90` - среднее число cart-days за 7 дней перед order-day.
68. `pre_order_search_lift_mean_90` - searches за 7 дней перед order / обычные 7 дней.
69. `pre_order_cart_lift_mean_90` - carts за 7 дней перед order / обычные 7 дней.
70. `last_order_had_search_same_day` - был ли search в день последней покупки.
71. `last_order_had_cart_same_day` - был ли cart в день последней покупки.
72. `last_order_searches_same_day` - searches в день последней покупки.
73. `last_order_cart_count_same_day` - to_cart в день последней покупки.
74. `journey_compactness_90` - same-day order journeys / all order days за 90.
75. `journey_long_intent_score_90` - pre-order search lift * order_gap_regularity_score.

## 6. Search vs Catalog Route Preference, 15

76. `search_route_gmv_share_90` - gmv_search / gmv за 90.
77. `cat_route_gmv_share_90` - gmv_cat / gmv за 90.
78. `route_balance_abs_90` - `abs(search_route_gmv_share_90 - cat_route_gmv_share_90)`.
79. `route_switch_count_120` - число смен доминирующего маршрута search/cat по активным дням.
80. `route_switch_rate_120` - `route_switch_count_120 / active_days_120`.
81. `last_route_is_search` - последняя активность была search-dominant.
82. `last_route_is_cat` - последняя активность была cat-dominant.
83. `route_entropy_120` - энтропия долей search/cat/both/no-route за 120 дней.
84. `search_route_recent_lift` - search_route_gmv_share_30 / search_route_gmv_share_180.
85. `cat_route_recent_lift` - cat_route_gmv_share_30 / cat_route_gmv_share_180.
86. `route_mixed_day_share_90` - доля дней, где были и search, и cat.
87. `route_single_day_share_90` - доля дней, где был только один маршрут.
88. `route_mixed_to_order_rate_90` - order rate в mixed-route дни.
89. `route_search_only_to_order_rate_90` - order rate в search-only дни.
90. `route_cat_only_to_order_rate_90` - order rate в cat-only дни.

## 7. Monetary Stability and Spikes, 15

91. `order_gmv_median_all` - медианный GMV в order-days.
92. `order_gmv_p25_all` - 25-й перцентиль GMV в order-days.
93. `order_gmv_p75_all` - 75-й перцентиль GMV в order-days.
94. `order_gmv_iqr_all` - p75 - p25 по GMV order-days.
95. `order_gmv_p90_all` - 90-й перцентиль GMV order-days.
96. `order_gmv_max_over_median` - максимальный order-day GMV / медиана.
97. `order_gmv_last_over_median` - GMV последнего order-day / медиана.
98. `order_gmv_last_over_p75` - GMV последнего order-day / p75.
99. `order_gmv_cv_all` - std / mean GMV order-days.
100. `order_gmv_mad_all` - median absolute deviation для GMV order-days.
101. `order_gmv_spike_count_180` - число order-days выше p75 всей истории за 180 дней.
102. `order_gmv_spike_share_180` - доля spike-order-days за 180 дней.
103. `order_gmv_concentration_top1_share` - самый большой order-day GMV / весь GMV.
104. `order_gmv_concentration_top3_share` - top-3 order-day GMV / весь GMV.
105. `monetary_stability_score` - `log1p(order_gmv_median_all) / (1 + order_gmv_cv_all)`.

## 8. Order Size and Repeat Buying, 15

106. `orders_per_order_day_all` - заказов на день с покупкой по всей истории.
107. `orders_per_order_day_90` - заказов на день с покупкой за 90 дней.
108. `orders_per_order_day_recent_lift` - `orders_per_order_day_90 / orders_per_order_day_all`.
109. `multi_order_day_count_all` - дней, где `to_ord > 1`.
110. `multi_order_day_share_all` - доля order-days с `to_ord > 1`.
111. `multi_order_day_share_90` - доля multi-order-days за 90 дней.
112. `last_order_day_order_count` - число заказов в последний день покупки.
113. `last_order_day_is_multi` - флаг, что последний order-day был multi-order.
114. `repeat_buy_density_30_after_first` - orders после первой покупки / дни после первой покупки в последние 30 доступных дней.
115. `repeat_buy_density_90_after_first` - orders после первой покупки / дни после первой покупки в последние 90 доступных дней.
116. `first_to_second_order_gap` - интервал между первой и второй покупкой.
117. `second_order_exists` - была ли повторная покупка.
118. `repeat_purchase_maturity` - `all_days_buy / max(tenure_days, 1)`.
119. `repeat_purchase_intensity_score` - `log1p(to_ord_sum_all) / (order_gap_median_all + 1)`.
120. `repeat_purchase_risk_score` - `order_cycle_phase * (1 - multi_order_day_share_all)`.

## 9. Post-Order Behavior Extended, 15

121. `post_order_gmv_search_sum` - gmv_search после последней покупки.
122. `post_order_gmv_cat_sum` - gmv_cat после последней покупки.
123. `post_order_search_to_cart_sum` - search_to_cart после последней покупки.
124. `post_order_search_to_ord_sum` - search_to_ord после последней покупки.
125. `post_order_cat_to_cart_sum` - cat_to_cart после последней покупки.
126. `post_order_cat_to_ord_sum` - cat_to_ord после последней покупки.
127. `post_order_cart_no_order_days` - cart-дни после последней покупки без order.
128. `post_order_search_no_cart_days` - search-дни после последней покупки без cart.
129. `post_order_route_entropy` - route entropy после последней покупки.
130. `post_order_active_velocity` - active days после покупки / days since last order.
131. `post_order_search_velocity` - searches после покупки / days since last order.
132. `post_order_cart_velocity` - carts после покупки / days since last order.
133. `post_order_intent_without_buy_score` - post-order cart/search pressure при отсутствии order.
134. `post_order_silent_buyer_flag` - покупал, но после последней покупки нет активности.
135. `post_order_returning_browser_flag` - покупал и после покупки продолжил search/cart.

## 10. Pre-Order Lead-Up, 15

136. `pre_last_order_searches_3d` - searches за 3 дня перед последней покупкой.
137. `pre_last_order_searches_7d` - searches за 7 дней перед последней покупкой.
138. `pre_last_order_carts_3d` - carts за 3 дня перед последней покупкой.
139. `pre_last_order_carts_7d` - carts за 7 дней перед последней покупкой.
140. `pre_last_order_active_days_7d` - активные дни за 7 дней перед последней покупкой.
141. `pre_last_order_search_lift_7d` - pre-order searches 7d / обычные 7d.
142. `pre_last_order_cart_lift_7d` - pre-order carts 7d / обычные 7d.
143. `pre_order_lead_time_search` - дней между последним search перед order и order.
144. `pre_order_lead_time_cart` - дней между последним cart перед order и order.
145. `pre_order_compact_flag` - search/cart/order уложились в 1 день.
146. `pre_order_warmup_flag` - search/cart были за 2-7 дней до order.
147. `pre_order_impulse_flag` - order без search/cart в последние 7 дней.
148. `pre_order_warmup_share_all` - доля order-days с warmup.
149. `pre_order_impulse_share_all` - доля impulse order-days.
150. `pre_order_planning_score` - warmup_share * median order GMV.

## 11. Last-N Event Sequence, 15

151. `last_active_event_type` - код доминирующего типа последнего активного дня.
152. `prev_active_event_type` - код доминирующего типа предпоследнего активного дня.
153. `last3_event_type_hash` - компактный hash последовательности последних 3 типов.
154. `last5_event_type_hash` - hash последних 5 типов.
155. `last3_contains_order` - была ли покупка в последних 3 active-days.
156. `last5_contains_cart_no_order` - был ли cart без order в последних 5 active-days.
157. `last5_search_only_count` - search-only дней в последних 5 active-days.
158. `last5_cat_only_count` - cat-only дней в последних 5 active-days.
159. `last5_cart_count` - cart-дней в последних 5 active-days.
160. `last5_order_count` - order-дней в последних 5 active-days.
161. `last5_searches_sum` - searches в последних 5 active-days.
162. `last5_gmv_sum` - GMV в последних 5 active-days.
163. `last_event_after_order_flag` - последняя активность была после последней покупки.
164. `last_event_cart_after_order_flag` - последняя активность cart после последней покупки.
165. `last_sequence_intent_score` - score по паттерну последних 5 active-days.

## 12. Dormancy and Revival, 15

166. `dormant_14_after_purchase_flag` - после покупки прошло 14+ дней без активности.
167. `dormant_30_after_purchase_flag` - после покупки прошло 30+ дней без активности.
168. `dormant_60_after_purchase_flag` - после покупки прошло 60+ дней без активности.
169. `revived_after_30d_silence_count` - число возвратов после паузы 30+ дней.
170. `revived_after_60d_silence_count` - число возвратов после паузы 60+ дней.
171. `last_revival_recency` - дней с последнего revival-события.
172. `last_revival_led_to_order` - привел ли последний revival к order в течение 7 дней.
173. `revival_to_order_rate_all` - доля revival, приводящих к order.
174. `revival_searches_mean` - среднее searches в день revival.
175. `revival_cart_rate` - доля revival-дней с cart.
176. `dormancy_depth` - `recency_days / (active_gap_p90_all + 1)`.
177. `dormancy_after_order_depth` - `recency_days / (order_gap_p90_all + 1)` для покупателей.
178. `sleeping_buyer_score` - order history strong, recent activity absent.
179. `reactivated_buyer_score` - был dormant, затем активен недавно.
180. `never_reactivated_flag` - были длинные паузы, но не было успешного revival.

## 13. Calendar Preference, 15

181. `weekend_active_share_all` - доля активных дней в выходные.
182. `weekend_search_share_all` - доля searches в выходные.
183. `weekend_cart_share_all` - доля carts в выходные.
184. `weekend_order_share_all` - доля order-days в выходные.
185. `weekend_gmv_share_all` - доля GMV в выходные.
186. `weekday_order_concentration` - max weekday order-days / all order-days.
187. `weekday_search_concentration` - max weekday searches / all searches.
188. `preferred_order_weekday` - weekday с максимумом order-days.
189. `preferred_search_weekday` - weekday с максимумом searches.
190. `last_order_weekday` - weekday последней покупки.
191. `last_active_weekday` - weekday последней активности.
192. `weekday_match_last_vs_preferred_order` - последний order weekday совпадает с preferred.
193. `weekend_recent_lift_90` - weekend order share 90 / all.
194. `weekday_entropy_orders` - энтропия покупок по дням недели.
195. `weekday_entropy_searches` - энтропия searches по дням недели.

## 14. Month-Phase and Payday Behavior, 15

196. `month_start_order_share` - доля order-days в дни 1-7 месяца.
197. `month_mid_order_share` - доля order-days в дни 8-20 месяца.
198. `month_end_order_share` - доля order-days в дни 21-конец месяца.
199. `month_start_gmv_share` - доля GMV в дни 1-7.
200. `month_end_gmv_share` - доля GMV в дни 21-конец.
201. `payday_around_10_order_share` - доля order-days в дни 8-12.
202. `payday_around_25_order_share` - доля order-days в дни 23-27.
203. `salary_cycle_order_score` - max(payday shares) / regular days share.
204. `last_order_month_phase` - фаза месяца последней покупки.
205. `last_active_month_phase` - фаза месяца последней активности.
206. `current_cutoff_month_phase` - фаза cutoff-даты, константа внутри cutoff.
207. `days_to_next_month_start` - дней от cutoff до начала следующего месяца.
208. `days_since_month_start_at_cutoff` - день месяца cutoff.
209. `month_phase_match_last_order` - cutoff phase совпадает с preferred order phase.
210. `month_phase_entropy_orders` - энтропия order-days по фазам месяца.

## 15. Cross-Sectional Rank Features, 15

211. `rank_pct_gmv_90` - percentile rank пользователя по GMV за 90 дней внутри cutoff.
212. `rank_pct_orders_90` - percentile rank по orders за 90 дней.
213. `rank_pct_searches_90` - percentile rank по searches за 90 дней.
214. `rank_pct_carts_90` - percentile rank по carts за 90 дней.
215. `rank_pct_active_days_90` - percentile rank по active days за 90 дней.
216. `rank_pct_recency_buy_inverse` - percentile rank по свежести покупки.
217. `rank_pct_recency_active_inverse` - percentile rank по свежести активности.
218. `rank_pct_order_gmv_median` - percentile rank по медианному order GMV.
219. `rank_pct_order_regularity` - percentile rank по regularity score.
220. `rank_pct_cart_friction` - percentile rank по cart friction score.
221. `rank_pct_search_intent` - percentile rank по search intent score.
222. `rank_pct_post_order_intent` - percentile rank по post-order intent.
223. `rank_pct_dormancy_depth` - percentile rank по dormancy depth.
224. `rank_pct_monetary_stability` - percentile rank по monetary stability.
225. `rank_pct_lifecycle_value_proxy` - percentile rank по composite lifecycle score.

## 16. Peer-Normalized Deviations, 15

226. `z_gmv_90` - z-score GMV 90 внутри cutoff.
227. `z_orders_90` - z-score orders 90.
228. `z_searches_90` - z-score searches 90.
229. `z_carts_90` - z-score carts 90.
230. `z_active_days_90` - z-score active days 90.
231. `z_order_gmv_median` - z-score медианного order GMV.
232. `z_order_gap_median` - z-score медианного order gap.
233. `z_recency_buy` - z-score recency_to_ord_days.
234. `z_recency_active` - z-score recency_days.
235. `rel_gmv_90_to_peer_median` - GMV 90 / median user GMV 90.
236. `rel_orders_90_to_peer_median` - orders 90 / median user orders 90.
237. `rel_searches_90_to_peer_median` - searches 90 / median user searches 90.
238. `rel_cart_friction_to_peer_median` - cart friction / median.
239. `rel_intent_score_to_peer_median` - search intent / median.
240. `peer_outlier_score` - сумма clipped z-score по money, orders, intent.

## 17. Nonlinear Interaction Scores, 15

241. `fresh_buyer_high_intent_score` - fresh order * recent search/cart pressure.
242. `stale_buyer_high_intent_score` - stale order * recent search/cart pressure.
243. `silent_high_value_score` - high historical GMV * no recent activity.
244. `active_low_purchase_score` - high activity * low purchase conversion.
245. `cart_heavy_no_order_score` - high carts * low recent orders.
246. `search_heavy_no_cart_score` - high searches * low carts.
247. `regular_buyer_overdue_score` - high regularity * overdue order cycle.
248. `irregular_big_spender_score` - high GMV * high order gap CV.
249. `weekend_buyer_now_weekend_score` - weekend preference * cutoff near weekend.
250. `payday_buyer_now_payday_score` - payday preference * cutoff near payday.
251. `post_order_browser_value_score` - post-order browsing * historical AOV.
252. `impulse_buyer_recent_search_score` - impulse share * recent search pressure.
253. `planner_buyer_recent_warmup_score` - planning score * recent warmup activity.
254. `dormant_reactivation_potential_score` - dormant but recent revival signals.
255. `route_preference_conflict_score` - recent route differs from historical preferred route.

## 18. Robust Trend and Shape Features, 15

256. `gmv_slope_weekly_120` - robust linear slope weekly GMV за 120 дней.
257. `orders_slope_weekly_120` - robust slope weekly orders.
258. `searches_slope_weekly_120` - robust slope weekly searches.
259. `carts_slope_weekly_120` - robust slope weekly carts.
260. `active_days_slope_weekly_120` - robust slope weekly active days.
261. `gmv_curvature_weekly_120` - последняя половина slope минус первая половина slope.
262. `orders_curvature_weekly_120` - curvature orders.
263. `searches_curvature_weekly_120` - curvature searches.
264. `carts_curvature_weekly_120` - curvature carts.
265. `gmv_trend_r2_weekly_120` - насколько хорошо weekly GMV ложится на линию.
266. `orders_trend_r2_weekly_120` - R2 weekly orders.
267. `searches_trend_r2_weekly_120` - R2 weekly searches.
268. `gmv_last_week_residual_120` - residual последней недели против robust trend.
269. `orders_last_week_residual_120` - residual последней недели orders.
270. `searches_last_week_residual_120` - residual последней недели searches.

## 19. Distribution and Entropy Features, 15

271. `daily_gmv_entropy_120` - энтропия распределения GMV по дням за 120.
272. `daily_orders_entropy_120` - энтропия orders по дням.
273. `daily_searches_entropy_120` - энтропия searches по дням.
274. `weekly_gmv_entropy_180` - энтропия GMV по неделям.
275. `weekly_orders_entropy_180` - энтропия orders по неделям.
276. `weekly_searches_entropy_180` - энтропия searches по неделям.
277. `gmv_top_week_share_180` - лучшая неделя GMV / весь GMV 180.
278. `orders_top_week_share_180` - лучшая неделя orders / all orders 180.
279. `searches_top_week_share_180` - лучшая неделя searches / all searches 180.
280. `active_top_week_share_180` - самая активная неделя / all active days 180.
281. `gmv_recent_week_rank_180` - ранг последней недели среди недель по GMV.
282. `orders_recent_week_rank_180` - ранг последней недели по orders.
283. `searches_recent_week_rank_180` - ранг последней недели по searches.
284. `activity_concentration_score` - среднее top-week share по activity/search/cart/order.
285. `spiky_vs_steady_score` - concentration score / regularity score.

## 20. Lifecycle Archetype Scores, 15

286. `new_high_intent_no_order_score` - новый пользователь, много search/cart, нет order.
287. `new_first_order_likely_score` - новый, свежий cart/search, конверсионные same-day сигналы.
288. `new_post_first_order_score` - недавно первая покупка и активность после нее.
289. `mature_regular_buyer_score` - long tenure, регулярные покупки, стабильный чек.
290. `mature_declining_buyer_score` - mature buyer, но активность/покупки просели.
291. `mature_reactivated_buyer_score` - mature buyer с revival после паузы.
292. `one_time_buyer_score` - одна покупка, потом тишина или слабая активность.
293. `window_shopper_score` - много search/catalog, мало cart/order.
294. `cart_abandoner_score` - много cart без order.
295. `catalog_browser_score` - доминирует catalog route без глубокого search.
296. `search_planner_score` - длинный pre-order search warmup.
297. `impulse_order_score` - order часто без search/cart warmup.
298. `high_value_at_risk_score` - высокий historical GMV, но dormancy growing.
299. `low_value_active_score` - активен, но низкий AOV/GMV.
300. `next_month_value_proxy` - composite из cadence, intent, monetary stability, dormancy и lifecycle.
