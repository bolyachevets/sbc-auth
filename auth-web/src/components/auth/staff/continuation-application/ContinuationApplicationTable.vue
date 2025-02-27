<template>
  <div id="continuation-application-review-table">
    <!-- Date submitted date range picker -->
    <DatePicker
      v-show="showDatePicker"
      ref="datePicker"
      :reset="dateRangeReset"
      class="mt-n4"
      @submit="updateDateRange($event)"
    />

    <v-form class="datatable-search continuation-review-search">
      <v-row
        dense
        class="row-margin"
      >
        <v-col
          sm="12"
          cols="6"
        >
          <transition name="slide-fade">
            <v-data-table
              :headers="headers"
              :items="reviews"
              :server-items-length="totalItemsCount"
              :options.sync="tableDataOptions"
              :disable-sort="true"
              :footer-props="{
                itemsPerPageOptions: paginationOptions
              }"
              hide-default-header
              fixed-header
              :loading="isTableLoading"
              :mobile-breakpoint="0"
              @update:options="updateItemsPerPage"
            >
              <!-- Displaying Status -->
              <template #[`item.status`]="{ item }">
                <div>
                  <span :class="['status-dot', item.status]" />
                  {{ displayStatus(item.status) }}
                </div>
              </template>

              <!-- Displaying Formatted Submission Date -->
              <template #[`item.submissionDate`]="{ item }">
                <div>{{ formatDate(item.submissionDate) }}</div>
              </template>

              <!-- Displaying NR Number and Tooltips-->
              <template #[`item.nrNumber`]="{ item }">
                <div>
                  {{ item.nrNumber }}
                  <IconTooltip
                    v-if="item.nrExpiryDate && daysLeft(item.nrExpiryDate) <= 14"
                    icon="mdi-alert"
                    maxWidth="300px"
                    colour="#F8661A"
                    :iconStyling="{'font-size': '1.5em', 'margin-left': '4px'}"
                    :location="{top: true}"
                  >
                    <div>
                      <strong>Alert:</strong><br>
                      <span>
                        The Name Request will expire
                        <span v-html="nrDaysLeftText(item.nrExpiryDate)" />.

                      </span>
                    </div>
                  </IconTooltip>
                  <IconTooltip
                    v-if="expiredDate(item.nrExpiryDate)"
                    icon="mdi-alert"
                    maxWidth="300px"
                    colour="#D3272C"
                    :iconStyling="{'font-size': '1.5em', 'margin-left': '4px'}"
                    :location="{top: true}"
                  >
                    <div>
                      <strong>Alert:</strong><br>
                      <span> This Name Request has expired.</span>
                    </div>
                  </IconTooltip>
                </div>
              </template>

              <!-- Loading -->
              <template #loading>
                <div
                  class="py-8 loading-datatable"
                >
                  Loading items...
                </div>
              </template>

              <!-- No data -->
              <template #no-data>
                <div
                  v-sanitize="noDataMessage"
                  class="py-8 no-data"
                />
              </template>

              <!-- Headers (two rows) -->
              <template #header="{}">
                <thead class="v-data-table-header">
                  <!-- First row has titles. -->
                  <tr class="header-row-1">
                    <th
                      v-for="(header, i) in headers"
                      :key="getIndexedTag('find-header-row', i)"
                      :scope="getIndexedTag('find-header-col', i)"
                      :class="{
                        'font-weight-bold': true,
                        'nr-number-column': header.value === 'nrNumber',
                        'submission-date-column': header.value === 'submissionDate',
                        'identifier-column': header.value === 'identifier'
                      }"
                      :style="reviewParams.sortBy === header.value ? 'color: black' : ''"
                      @click="!['action'].includes(header.value) ? changeSort(header.value) : null"
                    >
                      {{ header.text }}
                      <v-icon
                        v-if="!['action'].includes(header.value) && reviewParams.sortBy === header.value"
                        small
                        :style="reviewParams.sortBy === header.value ? 'color: black' : ''"
                      >
                        {{ reviewParams.sortDesc ? 'mdi-arrow-down' : 'mdi-arrow-up' }}
                      </v-icon>
                    </th>
                  </tr>

                  <!-- Second row has search boxes. -->
                  <tr class="header-row-2 mt-2 px-2">
                    <th
                      v-for="(header, i) in headers"
                      :key="getIndexedTag('find-header-row2', i)"
                      :scope="getIndexedTag('find-header-col2', i)"
                      :class="{
                        'nr-number-column': header.value === 'nrNumber',
                        'submission-date-column': header.value === 'submissionDate',
                        'identifier-column': header.value === 'identifier'
                      }"
                    >
                      <v-text-field
                        v-if="!['action','status','submissionDate'].includes(header.value)"
                        :id="header.value"
                        v-model.trim="reviewParams[header.value]"
                        input
                        type="search"
                        autocomplete="off"
                        class="text-input-style"
                        filled
                        :placeholder="header.text"
                        dense
                        hide-details="auto"
                      >
                        <!-- Clear icon -->
                        <template #append>
                          <v-icon
                            v-if="reviewParams[header.value]"
                            color="primary"
                            :style="{ marginTop: '-2px' }"
                            @click="clearOneParam(header.value)"
                          >
                            mdi-close
                          </v-icon>
                        </template>
                      </v-text-field>

                      <!-- Date Picker to select date submitted range -->
                      <div v-else-if="['submissionDate'].includes(header.value)">
                        <v-tooltip
                          bottom
                        >
                          <template #activator="{ on, attrs }">
                            <v-text-field
                              v-bind="attrs"
                              :value="truncatedDateRange(reviewParams.startDate, reviewParams.endDate)"
                              filled
                              :placeholder="'Date Submitted'"
                              readonly
                              dense
                              hide-details="auto"
                              v-on="on"
                              @click="showDatePicker = true"
                            >
                              <template #append>
                                <v-icon
                                  v-if="reviewParams.startDate"
                                  color="primary"
                                  :style="{ marginTop: '-2px' }"
                                  @click="updateDateRange({ startDate: '', endDate: '' })"
                                >
                                  mdi-close
                                </v-icon>
                                <v-icon
                                  color="primary"
                                  :style="{ marginTop: '-2px' }"
                                  @click="showDatePicker = true"
                                >
                                  mdi-calendar
                                </v-icon>
                              </template>
                            </v-text-field>
                          </template>
                          {{ fullDateRange(reviewParams.startDate, reviewParams.endDate) }}
                        </v-tooltip>
                      </div>

                      <!-- Drop down menu to select statuses -->
                      <div
                        v-else-if="['status'].includes(header.value)"
                        class="mt-0"
                      >
                        <v-select
                          v-model="reviewParams[header.value]"
                          :placeholder="reviewParams[header.value].length > 0 ? '' : 'Status'"
                          :items="statusTypes"
                          filled
                          item-text="text"
                          item-value="value"
                          multiple
                          attach
                          close-on-content-click="false"
                          data-test="select-status"
                          v-bind="$attrs"
                          hide-details="auto"
                          clearable
                          v-on="$listeners"
                        >
                          <template #selection="{ item, index }">
                            <!-- Display "Multiple Selected" if multiple statuses are selected -->
                            <v-tooltip
                              v-if="reviewParams[header.value].length > 1 && index === 0"
                              bottom
                            >
                              <template #activator="{ on, attrs }">
                                <span
                                  v-bind="attrs"
                                  v-on="on"
                                >
                                  Multiple Selected
                                </span>
                              </template>
                              <span>
                                <!-- when hover over 'Multiple Selected' showing what are selected. -->
                                {{ formattedStatuses() }}
                              </span>
                            </v-tooltip>
                            <!-- Display the item text if only one is selected -->
                            <span v-else-if="reviewParams[header.value].length === 1">
                              {{ item.text }}
                            </span>
                          </template>
                        </v-select>
                      </div>

                      <v-btn
                        v-else-if="searchParamsExist && header.value === 'action'"
                        outlined
                        color="primary"
                        class="action-btn clear-filter-button"
                        @click="clearSearchParams()"
                      >
                        <span class="clear-filter cursor-pointer">
                          Clear Filters
                          <v-icon
                            small
                            color="primary"
                          >mdi-close</v-icon>
                        </span>
                      </v-btn>
                    </th>
                  </tr>
                </thead>
              </template>

              <!-- Item Actions -->
              <template #[`item.action`]="{ item }">
                <div class="actions text-right">
                  <span class="open-action">
                    <v-btn
                      color="primary"
                      class="open-action-btn"
                      :class="{ 'active': item.nrNumber }"
                      :data-test="getIndexedTag('view-continuation-button', item.id)"
                      @click="view(item.id)"
                    >
                      {{ getButtonLabel(item.status) }}
                    </v-btn>
                  </span>
                  <!-- More Actions Menu -->
                  <span
                    v-if="item.nrNumber"
                  >
                    <v-menu
                      v-model="dropdown[item.id]"
                      offset-y
                      nudge-left="158"
                    >
                      <template #activator="{ on }">
                        <v-btn
                          color="primary"
                          class="more-actions-btn"
                          v-on="on"
                        >
                          <v-icon>{{ dropdown[item.nrNumber] ? 'mdi-menu-up' : 'mdi-menu-down' }}</v-icon>
                        </v-btn>
                      </template>
                      <v-list>
                        <v-list-item @click="goToNameXUI(item.nrNumber)">
                          <v-list-item-subtitle>
                            <v-icon style="font-size: 14px">mdi-format-list-bulleted</v-icon>
                            <span class="pl-2">Open Name Request</span>
                          </v-list-item-subtitle>
                        </v-list-item>
                      </v-list>
                    </v-menu>
                  </span>
                </div>
              </template>
            </v-data-table>
          </transition>
        </v-col>
      </v-row>
    </v-form>
  </div>
</template>

<script lang="ts">
import { ContinuationReviewIF, ReviewFilterParams } from '@/models/continuation-review'
import {
  DEFAULT_DATA_OPTIONS,
  cachePageInfo,
  getAndPruneCachedPageInfo,
  getPaginationOptions,
  hasCachedPageInfo
} from '@/components/datatable/resources'
import { PropType, computed, defineComponent, reactive, ref, toRefs, watch } from '@vue/composition-api'
import BusinessService from '@/services/business.services'
import CommonUtils from '@/util/common-util'
import ConfigHelper from '@/util/config-helper'
import { DataOptions } from 'vuetify'
import { DatePicker } from '@/components'
import IconTooltip from '@/components/IconTooltip.vue'
import { SessionStorageKeys } from '@/util/constants'
import debounce from '@/util/debounce'
import { goToNameXUI } from '@/util/navigation'
import moment from 'moment'
import { useI18n } from 'vue-i18n-composable'

export default defineComponent({
  name: 'ContinuationApplicationTable',
  components: { DatePicker, IconTooltip },
  props: {
    reviewSessionStorageKey: {
      type: String as PropType<SessionStorageKeys>,
      default: SessionStorageKeys.ReviewSearchFilter
    },
    reviewPaginationNumberOfItemsKey: {
      type: String as PropType<SessionStorageKeys>,
      default: SessionStorageKeys.ReviewPaginationNumberOfItems
    },
    reviewPaginationOptionsKey: {
      type: String as PropType<SessionStorageKeys>,
      default: SessionStorageKeys.ReviewPaginationOptions
    }
  },
  setup (props) {
    const { t } = useI18n()
    const statusDisplayMap = {
      AWAITING_REVIEW: 'Awaiting Review',
      CHANGE_REQUESTED: 'Change Requested',
      RESUBMITTED: 'Resubmitted',
      REJECTED: 'Rejected',
      APPROVED: 'Approved',
      ABANDONED: 'Abandoned',
      CANCELLED: 'Cancelled'
    }

    const state = reactive({
      reviews: [] as Array<ContinuationReviewIF>,
      headers: [
        { text: 'Date Submitted', value: 'submissionDate' },
        { text: 'NR Number', value: 'nrNumber' },
        { text: 'Identifying Number', value: 'identifier' },
        { text: 'Contact Email', value: 'contactEmail' },
        { text: 'Status', value: 'status' },
        { text: 'Actions', value: 'action' }
      ],
      statusTypes: [
        { text: 'Awaiting Review', value: 'AWAITING_REVIEW' },
        { text: 'Change Requested', value: 'CHANGE_REQUESTED' },
        { text: 'Resubmitted', value: 'RESUBMITTED' },
        { text: 'Rejected', value: 'REJECTED' },
        { text: 'Approved', value: 'APPROVED' },
        { text: 'Abandoned', value: 'ABANDONED' },
        { text: 'Cancelled', value: 'CANCELLED' }
      ],
      formatDate: CommonUtils.formatDisplayDate,
      totalItemsCount: 0,
      tableDataOptions: {} as Partial<DataOptions>,
      isTableLoading: false,
      searchParamsExist: false,
      sortBy: 'submissionDate',
      sortDesc: false,
      showDatePicker: false,
      dropdown: [] as Array<boolean>,
      reviewParams: {
        startDate: '',
        endDate: '',
        nrNumber: '',
        identifier: '',
        contactEmail: '',
        status: ['AWAITING_REVIEW', 'CHANGE_REQUESTED', 'RESUBMITTED'],
        sortBy: 'submissionDate',
        sortDesc: false
      } as ReviewFilterParams
    })

    const paginationOptions = computed(() => getPaginationOptions())

    const debouncedOrgSearch = debounce(async function (page = 1, pageLimit = state.tableDataOptions.itemsPerPage) {
      try {
        state.isTableLoading = true
        const completeSearchParams: ReviewFilterParams = {
          ...state.reviewParams,
          page: page,
          limit: pageLimit
        }
        const searchReviewResp = await BusinessService.searchReviews(completeSearchParams)
        state.reviews = searchReviewResp.reviews
        state.totalItemsCount = searchReviewResp?.total || 0
      } catch (error) {
        console.error(error)
      } finally {
        state.isTableLoading = false
      }
    })

    // date picker stuff
    const dateRangeReset = ref(0)
    const updateDateRange = (val: { endDate?: string, startDate?: string }) => {
      state.showDatePicker = false
      if (!(val.endDate && val.startDate)) {
        val = { startDate: '', endDate: '' }
      }
      state.reviewParams.startDate = val.startDate
      state.reviewParams.endDate = val.endDate
    }

    const fullDateRange = (startDate, endDate) => {
      if (startDate && endDate) {
        return `${moment(startDate).format('MMMM D, YYYY')} - ${moment(endDate).format('MMMM D, YYYY')}`
      }
      return ''
    }
    const truncatedDateRange = (startDate, endDate) => {
      if (startDate && endDate) {
        return `${moment(startDate).format('MMM D')} - ${moment(endDate).format('MMM D')}`
      }
      return ''
    }

    function mounted () {
      state.tableDataOptions = DEFAULT_DATA_OPTIONS
      const reviewSearchFilter = ConfigHelper.getFromSession(props.reviewSessionStorageKey) || ''
      try {
        state.reviewParams = JSON.parse(reviewSearchFilter)
      } catch {
        // Do nothing, we have defaults for review searchParams.
      }
      const hasInfo = hasCachedPageInfo(props.reviewPaginationOptionsKey)
      if (hasInfo) {
        state.tableDataOptions = getAndPruneCachedPageInfo(props.reviewPaginationOptionsKey)
      }
    }

    watch(() => state.reviewParams, function (value) {
      state.searchParamsExist = doSearchParametersExist(value)
      state.tableDataOptions = { ...getAndPruneCachedPageInfo(props.reviewPaginationOptionsKey), page: 1 }
      setSearchFilterToStorage(JSON.stringify(value))
      debouncedOrgSearch()
    }, { deep: true })

    watch(() => state.tableDataOptions, function (val) {
      debouncedOrgSearch(val?.page, val?.itemsPerPage)
    }, { deep: true })

    function getButtonLabel (status: string) {
      const reviewStates = ['AWAITING_REVIEW', 'RESUBMITTED']
      return reviewStates.includes(status) ? 'Review' : 'Open'
    }

    async function view (reviewId: string) {
      this.$router.push(`/staff/continuation-review/${reviewId}`)
    }

    function clearSearchParams () {
      dateRangeReset.value++
      state.reviewParams = {
        ...state.reviewParams,
        startDate: '',
        endDate: '',
        nrNumber: '',
        identifier: '',
        contactEmail: '',
        status: []
      }
    }

    function clearOneParam (param: string) {
      state.reviewParams = {
        ...state.reviewParams,
        [param]: ''
      }
    }

    function getIndexedTag (tag: string, index: number): string {
      return `${tag}-${index}`
    }

    function updateItemsPerPage (options: DataOptions): void {
      cachePageInfo(options, props.reviewPaginationOptionsKey)
    }

    const noDataMessage = computed(() => {
      return t(
        state.searchParamsExist
          ? 'searchReviewsNoResult'
          : 'searchReviewStartMessage'
      )
    })

    function setSearchFilterToStorage (val: string): void {
      ConfigHelper.addToSession(props.reviewSessionStorageKey, val)
    }

    function doSearchParametersExist (params: ReviewFilterParams): boolean {
      return params.startDate.length > 0 ||
                params.nrNumber.length > 0 ||
                params.identifier.length > 0 ||
                params.contactEmail.length > 0 ||
                params.status.length > 0
    }

    // Method to display the status
    function displayStatus (status: string): string {
      return statusDisplayMap[status]
    }
    // Method to format dates
    const formatDate = (dateString) => {
      if (!dateString) {
        return '' // when no date, return empty string
      }
      return moment(dateString).format('MMMM D, YYYY') // Format like "May 5, 2024"
    }
    // Method to check if NR date has passed
    const expiredDate = (nrDate: string): boolean => {
      return nrDate && !moment(nrDate).isAfter(moment())
    }
    // Method to check NR days left
    const daysLeft = (nrDate: string): number => {
      const diffHours = moment(nrDate).diff(moment(), 'hours')
      if (diffHours > 0 && diffHours <= 24) {
        return 1
      } else {
        const diffDays = moment(nrDate).diff(moment(), 'days')
        return diffDays > 0 ? diffDays : 50
      }
    }
    // Method to display NR days left text
    const nrDaysLeftText = (nrDate: string): string => {
      const expiryDate = moment(nrDate).endOf('day')
      const currentDate = moment().endOf('day')
      if (expiryDate.isSame(currentDate, 'day')) {
        return '<b>today</b>'
      } else {
        const diffDays = moment(nrDate).diff(moment(), 'days')
        return `in ${diffDays} days`
      }
    }

    function formattedStatuses () : string {
      return state.reviewParams['status']
        .map(statusValue => state.statusTypes.find(status => status.value === statusValue)?.text)
        .join(', ')
    }

    function changeSort (column: string) {
      if (state.sortBy === column) {
        state.sortDesc = !this.sortDesc
      } else {
        state.sortBy = column
        state.sortDesc = false
      }
      state.reviewParams = {
        ...state.reviewParams,
        sortBy: state.sortBy,
        sortDesc: state.sortDesc
      }
    }

    mounted()

    return {
      ...toRefs(state),
      debouncedOrgSearch,
      dateRangeReset,
      view,
      clearSearchParams,
      changeSort,
      daysLeft,
      nrDaysLeftText,
      displayStatus,
      formatDate,
      fullDateRange,
      getIndexedTag,
      getButtonLabel,
      noDataMessage,
      setSearchFilterToStorage,
      truncatedDateRange,
      doSearchParametersExist,
      paginationOptions,
      goToNameXUI,
      expiredDate,
      updateItemsPerPage,
      updateDateRange,
      formattedStatuses,
      clearOneParam
    }
  }
})
</script>

<style lang="scss" scoped>
@import '@/assets/scss/theme.scss';
// Note this uses .datatable-search
@import '@/assets/scss/search.scss';

// Vuetify Override
.theme--light.v-list-item .v-list-item__action-text, .theme--light.v-list-item .v-list-item__subtitle {
  color: $app-blue;
  .v-icon.v-icon {
    color: $app-blue;
  }
}

// Class binding a vuetify override.
// To handle the sticky elements overlap in the custom scrolling data table.
.header-high-layer {
  ::v-deep {
    th {
      z-index: 2 !important;
    }
  }
}

::v-deep .theme--light.v-data-table .v-data-table__empty-wrapper {
  color: $gray7;
  &:hover {
    background-color: transparent;
  }
}

::v-deep .v-input__icon--clear .v-icon {
  margin-top: -2px;
  color: var(--v-primary-base);
}

.business-dashboard-icon {
  opacity: 1;
  width: 14px;
  height: 14px;
  transform: rotate(0deg);
  display: inline-flex;
}

.continuation-review-search {
  table > thead > tr > th {
    width: 210px !important;
    min-width: 210px !important;
  }

  .nr-number-column {
    width: 140px !important;
    min-width: 140px !important;
  }

  .submission-date-column {
    width: 180px !important;
    min-width: 180px !important;
  }

  .identifier-column {
    width: 180px !important;
    min-width: 180px !important;
  }

  .open-action-btn,
  .more-actions-btn {
    box-shadow: 0 1px 1px 0px rgb(0 0 0 / 20%), 0 2px 2px 0 rgb(0 0 0 / 14%), 0 1px 5px 0 rgb(0 0 0 / 12%);
    -webkit-box-shadow: 0 1px 1px 0px rgb(0 0 0 / 20%), 0 2px 2px 0 rgb(0 0 0 / 14%), 0 1px 5px 0 rgb(0 0 0 / 12%);
  }

  .open-action-btn {
    min-width: 7rem !important;
    &.active {
      border-top-right-radius: 0;
      border-bottom-right-radius: 0;
      min-width: 5rem !important;
    }
  }

  .more-actions-btn {
    padding-left: 0px;
    padding-right: 0px;
    border-top-left-radius: 0;
    border-bottom-left-radius: 0;
    max-width: 30px !important;
    min-width: 30px !important;
    margin-left: 0.05rem;
  }

  // Inline for Clear Filters
  .clear-filter-button {
    margin-top: 2px;
    padding: 7px !important;
  }

  .clear-filter {
    line-height: 1.5;
  }

  // As requested by Tracey.
  ::v-deep input, ::v-deep .v-select__selection {
    color:#212529 !important;
  }

  ::v-deep ::placeholder{
    color:#495057 !important;
  }

  .v-data-table th {
    font-size: 0.75rem;
  }

  ::v-deep .v-data-footer {
    min-width: 100%;
  }

  .no-data, .loading-datatable {
    border: 0px;
    position: sticky;
    width: 1230px;
    left: 0;
    flex-grow: 0;
    flex-shrink: 0;
  }

  ::v-deep .v-data-table__wrapper::-webkit-scrollbar {
    width: .625rem;
    height: 0.50rem;
    overflow-x: hidden;
  }

  ::v-deep .v-data-table__wrapper::-webkit-scrollbar-track {
    overflow: auto;
  }

  ::v-deep .v-data-table__wrapper::-webkit-scrollbar-thumb {
    border-radius: 5px;
    background-color: lightgray;
  }

  table > thead > tr > th:last-child {
    width: 130px !important;
    min-width: 130px !important;
  }

  // The TD cells don't seem to be scoped properly.
  // Thus the usage of v-deep.
  ::v-deep tr:hover td:last-child {
    background-color: inherit !important;
  }

  table > thead > tr > th:last-child,
  ::v-deep table > tbody > tr > td:last-child:not([colspan]) {
    position: sticky !important;
    position: -webkit-sticky !important;
    right: 0;
    z-index: 1;
    background: white;
    text-align: right !important;
  }

 ::v-deep table > tbody > tr > td {
    border: 0px;
    padding-top: 0.5rem;
    padding-bottom: 0.5rem;
  }

  ::v-deep table > tbody > tr > td:last-child:not([colspan]) {
    padding-left: 3px !important;
    padding-right: 3px !important;
  }
}

.bottom-tooltip:before {
  content: ' ';
  position: absolute;
  top: -20px;
  left: 50%;
  margin-left: -10px;
  width: 20px;
  height: 20px;
  border-width: 10px 10px 10px 10px;
  border-style: solid;
  border-color: transparent transparent rgba(50, 50, 50, 0.7) transparent;
}

.status-dot {
  height: 8px;
  width: 8px;
  border-radius: 50%;
  display: inline-block;
  margin-right: 8px;

  &.AWAITING_REVIEW { background-color: $app-alert-orange; }
  &.CHANGE_REQUESTED { background-color: $app-blue; }
  &.RESUBMITTED { background-color: $app-red; }
  &.REJECTED,
  &.APPROVED,
  &.ABANDONED,
  &.CANCELLED { background-color: $gray3; }
}
</style>
