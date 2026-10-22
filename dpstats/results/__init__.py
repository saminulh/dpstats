"""Scientific outcomes, current-Result acquisition, and their persisted Attempt records."""

from dpstats.results._outcomes import (
    ResultAttempt as ResultAttempt,
    ResultStatus as ResultStatus,
)
from dpstats.results._release import (
    answered_coordinates as answered_coordinates,
    read_result_row_count as read_result_row_count,
    result_files as result_files,
)
from dpstats.results._frame import results_frame as results_frame
from dpstats.results._partitions import (
    PartitionCensus as PartitionCensus,
    compact_partition as compact_partition,
    fragmented_partitions as fragmented_partitions,
    partition_census as partition_census,
)
